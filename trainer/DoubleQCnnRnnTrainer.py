import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Deque, Tuple, List
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging
import torch.nn.functional as F

from model.GoldMiner import GoldMinerEnv
from agent.QCnnRnn.QCnnRnn import QCnnRnn
from agent.QCnnRnn.Embedder import Embedder
from define import set_ai_action_info

# 50 actions - chia đều góc 15-165° thành 50 khoảng (3° mỗi khoảng)
angle_bins = [
    (15.0, 18.0),    # Action 0
    (18.0, 21.0),    # Action 1
    (21.0, 24.0),    # Action 2
    (24.0, 27.0),    # Action 3
    (27.0, 30.0),    # Action 4
    (30.0, 33.0),    # Action 5
    (33.0, 36.0),    # Action 6
    (36.0, 39.0),    # Action 7
    (39.0, 42.0),    # Action 8
    (42.0, 45.0),    # Action 9
    (45.0, 48.0),    # Action 10
    (48.0, 51.0),    # Action 11
    (51.0, 54.0),    # Action 12
    (54.0, 57.0),    # Action 13
    (57.0, 60.0),    # Action 14
    (60.0, 63.0),    # Action 15
    (63.0, 66.0),    # Action 16
    (66.0, 69.0),    # Action 17
    (69.0, 72.0),    # Action 18
    (72.0, 75.0),    # Action 19
    (75.0, 78.0),    # Action 20
    (78.0, 81.0),    # Action 21
    (81.0, 84.0),    # Action 22
    (84.0, 87.0),    # Action 23
    (87.0, 90.0),    # Action 24
    (90.0, 93.0),    # Action 25
    (93.0, 96.0),    # Action 26
    (96.0, 99.0),    # Action 27
    (99.0, 102.0),   # Action 28
    (102.0, 105.0),  # Action 29
    (105.0, 108.0),  # Action 30
    (108.0, 111.0),  # Action 31
    (111.0, 114.0),  # Action 32
    (114.0, 117.0),  # Action 33
    (117.0, 120.0),  # Action 34
    (120.0, 123.0),  # Action 35
    (123.0, 126.0),  # Action 36
    (126.0, 129.0),  # Action 37
    (129.0, 132.0),  # Action 38
    (132.0, 135.0),  # Action 39
    (135.0, 138.0),  # Action 40
    (138.0, 141.0),  # Action 41
    (141.0, 144.0),  # Action 42
    (144.0, 147.0),  # Action 43
    (147.0, 150.0),  # Action 44
    (150.0, 153.0),  # Action 45
    (153.0, 156.0),  # Action 46
    (156.0, 159.0),  # Action 47
    (159.0, 162.0),  # Action 48
    (162.0, 165.0)   # Action 49
]


# ReplayBuffer cho QCnnRnn: mỗi sample là một episode (dict)
class ReplayBuffer:
    """
    Replay buffer cho QCnnRnn, mỗi sample là một episode dict gồm:
    - env_feats: torch.Tensor [max_steps, 10]
    - item_feats: torch.Tensor [max_steps, max_items, 23]
    - masks: torch.Tensor [max_steps, max_items]
    - actions: torch.Tensor [max_steps-1]
    - rewards: torch.Tensor [max_steps-1]
    - seq_len: int (số bước thực tế, không tính padding)
    """
    def __init__(self, capacity=1000, device='cpu'):
        self.buffer = []
        self.capacity = capacity
        self.device = device

    def push(self, env_feats, item_feats, masks, actions, rewards, seq_len):
        """
        Thêm một episode vào buffer
        Args:
            env_feats: torch.Tensor [max_steps, 10]
            item_feats: torch.Tensor [max_steps, max_items, 23]
            masks: torch.Tensor [max_steps, max_items]
            actions: torch.Tensor [max_steps]
            rewards: torch.Tensor [max_steps]
            seq_len: int
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        # Move tensors to device when pushing
        self.buffer.append((
            env_feats[:15].to(self.device), 
            item_feats[:15].to(self.device), 
            masks[:15].to(self.device), 
            actions[:15].to(self.device), 
            rewards[:15].to(self.device), 
            seq_len if seq_len <= 15 else 15
        ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        episodes = random.sample(self.buffer, batch_size)
        
        env_feats_list, item_feats_list, masks_list, actions_list, rewards_list, seq_lens_list = zip(*episodes)
        
        env_feats = torch.stack(env_feats_list)          # [B, max_steps, 10]
        item_feats = torch.stack(item_feats_list)        # [B, max_steps, max_items, 23]
        masks = torch.stack(masks_list)                  # [B, max_steps, max_items]
        actions = torch.stack(actions_list)              # [B, max_steps-1]
        rewards = torch.stack(rewards_list)              # [B, max_steps-1]
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int)            # [B]
        
        return {
            'env_feats': env_feats,
            'item_feats': item_feats,
            'masks': masks,
            'actions': actions,
            'rewards': rewards,
            'seq_lens': seq_lens
        }
        
def compute_td_loss_double_dqn(
    q_values,           # [B, T, A] - từ online agent
    target_q_values,    # [B, T, A] - từ target agent
    online_next_q,      # [B, T, A] - từ online agent cho next states (dùng để chọn action)
    actions,            # [B, T]
    rewards,            # [B, T]
    seq_lens,           # [B]
    gamma=0.99,
):
    """
    Double DQN: Dùng online network để chọn action, target network để đánh giá.
    actions[:, 0] = A (no-op / pad), actions[:, 1:] < A
    rewards[:, 0] = 0
    terminal index = seq_len - 1
    """

    device = q_values.device
    B, T, A = q_values.shape

    # ----- 1. Build transitions i: state_i -> action_{i+1} -> r_{i+1} -> state_{i+1} -----
    q_s_i        = q_values[:, :-1, :]           # [B, T-1, A]
    target_s_ip1 = target_q_values[:, 1:, :]     # [B, T-1, A]
    online_s_ip1 = online_next_q[:, 1:, :]       # [B, T-1, A]

    a_ip1 = actions[:, 1:]                       # [B, T-1]
    r_ip1 = rewards[:, 1:]                       # [B, T-1]

    # ----- 2. Mask theo seq_lens -----
    time_idx   = torch.arange(T - 1, device=device).unsqueeze(0)     # [1, T-1]
    valid_mask = time_idx < (seq_lens.unsqueeze(1) - 1)              # [B, T-1]

    next_idx         = time_idx + 1
    next_is_terminal = next_idx == (seq_lens.unsqueeze(1) - 1)       # [B, T-1]

    if not valid_mask.any():
        return q_values.new_tensor(0.0)

    # ----- 3. Q(s_i, a_{i+1}) từ online -----
    a_ip1_clamped = a_ip1.clamp(0, A - 1)
    q_sa = q_s_i.gather(-1, a_ip1_clamped.unsqueeze(-1)).squeeze(-1)   # [B, T-1]

    # ----- 4. Double DQN Target: r_{i+1} + gamma * Q_target(s_{i+1}, argmax_a Q_online(s_{i+1}, a)) -----
    with torch.no_grad():
        # Chọn action bằng online network
        best_actions = online_s_ip1.argmax(dim=-1)                    # [B, T-1]
        # Đánh giá action bằng target network
        next_q_double = target_s_ip1.gather(-1, best_actions.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        gamma_term = gamma * next_q_double * (~next_is_terminal).float()
        target = r_ip1 + gamma_term                                   # [B, T-1]

    # ----- 5. Mask & loss -----
    q_sa_valid     = q_sa[valid_mask]
    target_valid   = target[valid_mask]
    loss = F.smooth_l1_loss(q_sa_valid, target_valid)
    return loss

class DoubleQCnnRnnTrainer:
    """Trainer cho Double Deep Q-Learning"""
    
    def __init__(
        self,
        env: GoldMinerEnv,
        agent: QCnnRnn,
        lr: float = 1e-4,
        gamma: float = 1.0,
        epsilon_start: float = 0.5,  # Thấp hơn vì chỉ có 2 actions
        epsilon_end: float = 0.01,    # End sớm hơn
        epsilon_decay: float = 0.99,  # Decay nhanh hơn (cho Exponentially), replay buffer đã giúp break correlation
        buffer_size: int = 320,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        train_freq: int = 1,  # Tần suất training: train mỗi train_freq steps
        num_planning: int = 1,  # Số lần quét buffer (planning) hoặc số batches (standard)
        use_planning: bool = True,  # True: planning approach, False: standard DQN
        warmup_steps: int = 1000,  # Số steps warmup với random actions trước khi train
        explore_strategy: str = 'Exponentially',  # 'Linearly' or 'Exponentially'
        num_episodes: int = 1000,  # Tổng số episodes (dùng cho Linearly decay)
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.agent = agent.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.num_planning = num_planning
        self.use_planning = use_planning
        self.warmup_steps = warmup_steps
        self.explore_strategy = explore_strategy
        self.num_episodes = num_episodes
        
        # Target network
        self.target_agent = QCnnRnn(
            d_model=agent.d_model,
            n_actions=agent.n_actions,
            d_hidden=agent.d_hidden,
        ).to(device)
        self.update_target_network()
        self.target_agent.eval()
        
        # Optimizer và loss
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, device=device)
        
        # Tracking
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Logger (will be setup in train())
        self.logger = None
        
    def warmup_buffer(self, warmup_path: str):
        print(f"\n[3.5/4] Loading warmup buffer...")
        print(f"  Warmup buffer: {warmup_path}")
        import pickle
        with open(warmup_path, 'rb') as f:
            buffer_data = pickle.load(f)
            
            
            for epi in buffer_data:
                self.replay_buffer.push(epi['env_feats'], epi['item_feats'], epi['masks'], epi['actions'], epi['rewards'], epi['seq_len'])
            
        print(f"✓ Warmup buffer loaded")
        print(f"  Loaded {len(buffer_data)} transitions")
        print(f"  Buffer size: {len(self.replay_buffer)}")
        
    def setup_logger(self, log_file: str = 'training.log'):
        """Setup logger to write to file"""
        self.logger = logging.getLogger('DQNTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler (chỉ log vào file, không ra terminal)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        
    def log(self, message: str):
        """Log message to file only"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
        
    def update_target_network(self):
        """Copy weights từ agent sang target_agent"""
        self.target_agent.load_state_dict(self.agent.state_dict())
    
    @torch.no_grad()
    def select_action(self, state: dict, prev_action: int, miss_streak: int, hidden_state: torch.Tensor, training: bool = True) -> tuple:
        """
        Chọn action với epsilon-greedy policy
        
        Args:
            state: Game state dict
            miss_streak: Số lần miss liên tiếp
            training: Nếu True thì dùng epsilon-greedy, False thì greedy
            
        Returns:
            (action, used_model, new_hidden_state): action được chọn và flag cho biết có dùng model không
        """
        # Preprocess state for QCNN
        env_feats, item_feats, mask = Embedder.preprocess_state(state, return_batch=True)
        
        # Convert to device
        env_feats = env_feats.unsqueeze(0).to(self.device)  # [B, T, 10]
        item_feats = item_feats.unsqueeze(0).to(self.device)  # [B, T, max_items, 23]
        mask = mask.unsqueeze(0).to(self.device)  # [B, T, max_items]
        
        q_values, new_hidden_state = self.agent.predict_step(env_feats, item_feats, mask, torch.tensor(prev_action, device=self.device).reshape(1, 1), torch.tensor(miss_streak, device=self.device).reshape(1, 1), hidden_state)
        # Warmup phase: chỉ dùng random actions
        if training and len(self.replay_buffer) < self.warmup_steps:
            action = random.randint(0, self.agent.n_actions - 1)
            used_model = False
            q_value = None
        # Epsilon-greedy action selection
        elif training and random.random() < self.epsilon:
            action = random.randint(0, self.agent.n_actions - 1)
            used_model = False
            q_value = None
        else:
            with torch.no_grad():
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0][action].cpu().item()
            used_model = True
        
        # Lưu thông tin action vào global state để hiển thị trên màn hình
        set_ai_action_info(action, q_value, used_model)
        
        return action, used_model, new_hidden_state
    
    def train_step(self, cur_step) -> list:
        """
        Training step với 2 modes:
        - Planning approach: Quét qua toàn bộ buffer num_planning lần
        - Standard DQN: Sample num_planning batches ngẫu nhiên
        
        Args:
            cur_step: Current step number (for logging)
        
        Returns:
            losses: List of losses
        """
        if len(self.replay_buffer) == 0:
            return []
        
        if self.use_planning:
            return self._train_step_planning(cur_step)
        else:
            return self._train_step_standard(cur_step)
    
    def _train_step_planning(self, cur_step) -> list:
        """
        Planning approach: Quét qua toàn bộ buffer num_planning lần.
        """
        losses = []
        
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        # Quét qua buffer num_planning lần
        for planning_iter in range(self.num_planning):
            iter_loss = 0.0
            num_batches = 0
            
            # Lấy tất cả transitions và shuffle
            all_episodes = list(self.replay_buffer.buffer)
            random.shuffle(all_episodes)
            
            # Quét qua toàn bộ buffer theo batches
            for batch_start in range(0, len(all_episodes), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(all_episodes))
                
                if batch_end - batch_start < self.batch_size:
                    continue  # Skip batch cuối nếu không đủ size
                
                # Lấy batch và unpack
                batch_episodes = all_episodes[batch_start:batch_end]
                env_feats_list, item_feats_list, mask_list, action_list, reward_list, seq_len_list = zip(*batch_episodes)
                
                # Stack thành batch dict
                batch = {
                    'env_feats': torch.stack(env_feats_list),          # [B, max_steps, 10]
                    'item_feats': torch.stack(item_feats_list),        # [B, max_steps, max_items, 23]
                    'masks': torch.stack(mask_list),                    # [B, max_steps, max_items]
                    'actions': torch.stack(action_list),               # [B, max_steps-1]
                    'rewards': torch.stack(reward_list),               # [B, max_steps-1]
                    'seq_lens': torch.tensor(seq_len_list, dtype=torch.int)            # [B]
                }
                
                loss = self._train_on_batch(batch)
                iter_loss += loss.item()
                num_batches += 1
            
            # Tính avg loss cho lần quét này
            avg_iter_loss = iter_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_iter_loss)
            
            # Log loss cho mỗi lần quét
            self.log(f"Step {cur_step}  Planning {planning_iter+1}/{self.num_planning} - Avg Loss: {avg_iter_loss:.8f} - Batches: {num_batches} - Buffer: {len(self.replay_buffer)}")
        
        return losses
    
    def _train_step_standard(self, cur_step) -> list:
        """
        Standard DQN: Sample num_planning batches ngẫu nhiên và train.
        """
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        losses = []
        
        # Sample và train trên num_planning batches
        for batch_idx in range(self.num_planning):
            batch = self.replay_buffer.sample(self.batch_size)
            
            loss = self._train_on_batch(batch)
            losses.append(loss)
            
            # Log loss cho mỗi batch
            if cur_step % 3 == 0:
                self.log(f"Step {cur_step}  Batch {batch_idx+1}/{self.num_planning} - Loss: {loss:.8f} - Buffer: {len(self.replay_buffer)}")
        
        return losses
    
    def _train_on_batch(self, batch: dict) -> float:
        """
        Train trên một batch và trả về loss.
        
        Args:
            batch: Dict từ buffer.sample() - TẤT CẢ ĐÃ LÀ TORCH TENSORS (CPU)
                   Chỉ cần .to(device) để chuyển sang GPU
            
        Returns:
            loss: Loss value
        """
        # Move tensors from CPU to device (GPU/CPU)
        env_feats = batch['env_feats'].to(self.device)  # [B, max_steps, 10]
        item_feats = batch['item_feats'].to(self.device)  # [B, max_steps, max_items, 23]
        mask = batch['masks'].to(self.device)  # [B, max_steps, max_items]
        
        actions = batch['actions'].to(self.device)  # [B, max_steps]
        rewards = batch['rewards'].to(self.device)  # [B, max_steps]
        seq_lens = batch['seq_lens'].to(self.device)  # [B]
        
        # Compute Q(s, a) for online agent
        x = self.agent.get_gru_input(env_feats, item_feats, mask, actions)
        o, h = self.agent.gru(x)
        q_values = self.agent.predictor(o)  # [B, max_steps, n_actions]
        
        # Compute Q values for target network
        with torch.no_grad():
            next_x = self.target_agent.get_gru_input(env_feats, item_feats, mask, actions)
            next_o, next_h = self.target_agent.gru(next_x)
            target_q_values = self.target_agent.predictor(next_o)  # [B, max_steps, n_actions]
        
        # Compute loss using Double DQN (online network selects action, target network evaluates)
        loss = compute_td_loss_double_dqn(
            q_values=q_values, 
            target_q_values=target_q_values,
            online_next_q=q_values,  # Use online network to select best action
            actions=actions, 
            rewards=rewards, 
            seq_lens=seq_lens, 
            gamma=self.gamma
        )
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self) -> Tuple[float, int]:
        """
        Train 1 episode
        
        Returns:
            total_reward: Tổng reward trong episode
            steps: Số steps trong episode
        """
        state, _ = self.env.reset()        
        # Storage for this episode
        env_feats_list = []
        item_feats_list = []
        masks_list = []
        actions_list = []
        rewards_list = []
        
        action_buffer = len(angle_bins)
        angle_decision = None
        done = False
        reward_buffer = 0.0
        miss_streak = 0  # Số lần miss liên tiếp
        prev_num_items = -1
        prev_total_points = 0.0  # Tổng point của các items để detect TNT
        hidden_state = torch.zeros(1, self.agent.num_layers, self.agent.d_hidden).to(self.device) # batch first
        
        while True:
            if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
            else:
                if angle_decision is None or done:
                    env_f, item_f, mask = Embedder.preprocess_state(state)
                    env_feats_list.append(env_f)
                    item_feats_list.append(item_f)
                    masks_list.append(mask)
                    cur_num_items = mask.sum().item()
                    
                    # Tính tổng point của các items hiện tại
                    cur_total_points = sum(item['point'] for item in state['items'])
                    
                    # Phát hiện TNT explosion: reward = 0 nhưng tổng point giảm
                    if reward_buffer == 0 and cur_total_points < prev_total_points:
                        lost_points = prev_total_points - cur_total_points
                        tnt_penalty = -self.env.c_tnt * lost_points / self.env.reward_scale
                        reward_buffer += tnt_penalty
                    
                    if cur_num_items == prev_num_items:
                        # Miss: tăng streak và áp dụng penalty tuyến tính
                        miss_streak += 1
                        miss_penalty = -self.env.c_miss * miss_streak / self.env.reward_scale
                        reward_buffer += miss_penalty
                    else:
                        # Hit: reset streak
                        miss_streak = 0
                    prev_num_items = cur_num_items
                    prev_total_points = cur_total_points
                    
                    # Save accumulated reward r_t from previous action (if exists)
                    rewards_list.append(reward_buffer)
                    actions_list.append(action_buffer)
                    
                    if done:
                        break
                    
                    action_buffer, used_model, hidden_state = self.select_action(state, action_buffer, miss_streak, hidden_state, training=True)
                    angle_decision = angle_bins[action_buffer]
                    reward_buffer = 0
                
                current_angle = state['rope_state']['direction']
                if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                    next_state, reward, terminated, truncated, info = self.env.step(1)  # Fire action
                    angle_decision = None
                else:
                    next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
    
            done = terminated or truncated
            reward_buffer += reward
            state = next_state
        
        
        # Pad episode to max_steps
        seq_len = len(env_feats_list)
        
        if seq_len > 0:
            # Stack states
            env_feats_tensor = torch.stack(env_feats_list)  # [seq_len, 10]
            item_feats_tensor = torch.stack(item_feats_list)  # [seq_len, max_items, 23]
            masks_tensor = torch.stack(masks_list)  # [seq_len, max_items]
            
            # Convert actions and rewards to tensors
            actions_tensor = torch.tensor(actions_list, dtype=torch.long)  # [seq_len]
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)  # [seq_len]
            
            # Pad to max_steps
            if seq_len < 15:
                pad_len = 15 - seq_len
                
                # Pad env_feats
                env_pad = torch.zeros(pad_len, 10, dtype=torch.float32)
                env_feats_tensor = torch.cat([env_feats_tensor, env_pad], dim=0)
                
                # Pad item_feats
                item_pad = torch.zeros(pad_len, 30, 23, dtype=torch.float32)
                item_feats_tensor = torch.cat([item_feats_tensor, item_pad], dim=0)
                
                # Pad masks
                mask_pad = torch.zeros(pad_len, 30, dtype=torch.float32)
                masks_tensor = torch.cat([masks_tensor, mask_pad], dim=0)
                
                # Pad actions max_steps
                action_pad_len = 15 - len(actions_list)
                if action_pad_len > 0:
                    action_pad = torch.zeros(action_pad_len, dtype=torch.long)
                    actions_tensor = torch.cat([actions_tensor, action_pad], dim=0)
                
                # Pad rewards
                reward_pad_len = 15 - len(rewards_list)
                if reward_pad_len > 0:
                    reward_pad = torch.zeros(reward_pad_len, dtype=torch.float32)
                    rewards_tensor = torch.cat([rewards_tensor, reward_pad], dim=0)
            
            # Save episode
            self.replay_buffer.push(env_feats_tensor, item_feats_tensor, masks_tensor, actions_tensor, rewards_tensor, seq_len)
            episode_data = {
                'env_feats': env_feats_tensor,  # [max_steps, 10]
                'item_feats': item_feats_tensor,  # [max_steps, max_items, 23]
                'masks': masks_tensor,  # [max_steps, max_items]
                'actions': actions_tensor,  # [max_steps-1]
                'rewards': rewards_tensor,  # [max_steps-1]
                'seq_len': seq_len  # Actual length before padding
            }
        
        return rewards_tensor.sum().item(), seq_len
    
    def train(
        self,
        num_episodes: int,
        save_freq: int = 100,
        eval_freq: int = 50,
        eval_episodes: int = 5,
        save_dir: str = 'checkpoints',
        log_file: str = 'training_log.json',
        training_log_file: str = 'training.log'
    ):
        """
        Train agent cho num_episodes
        
        Args:
            num_episodes: Số episodes để train
            save_freq: Tần suất lưu checkpoint (episodes)
            eval_freq: Tần suất evaluation (episodes)
            eval_episodes: Số episodes cho mỗi lần eval
            save_dir: Thư mục lưu checkpoints
            log_file: File log training metrics (JSON)
            training_log_file: File log quá trình training (text)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logger
        self.setup_logger(training_log_file)
        
        self.log(f"Starting training for {num_episodes} episodes")
        self.log(f"Device: {self.device}")
        self.log(f"Warmup steps: {self.warmup_steps} (random actions only)")
        self.log(f"Replay buffer size: {len(self.replay_buffer)}")
        self.log("-" * 60)
        
        for episode in range(1, num_episodes + 1):         
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()
                self.log(f"Updated target network at step {self.total_steps}")

            if len(self.replay_buffer) >= self.warmup_steps:
                self.log(f'making agent better...')
                self.train_step(cur_step=episode)
            self.log(f"\nStarting Episode {episode}/{num_episodes}")
            
            # Train episode
            episode_reward, episode_steps = self.train_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.total_steps += 1
            
            # Decay epsilon
            if self.explore_strategy == 'Linearly':
                # Linear decay: epsilon = start - (start - end) * (episode / total_episodes)
                self.epsilon = max(self.epsilon_end, 
                                  self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (episode / num_episodes))
            else:  # Exponentially
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # In điểm kiếm được sau mỗi episode
            buffer_size = len(self.replay_buffer)
            self.log(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps} | Buffer: {buffer_size}")
            print(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps} | Buffer: {buffer_size}")
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0.0
                warmup_status = "WARMUP" if self.replay_buffer.__len__() < self.warmup_steps else "TRAINING"
                
                self.log(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.3f} | "
                      f"Avg(10): {avg_reward:.8f} | "
                      f"Steps: {episode_steps} | "
                      f"Loss: {avg_loss:.8f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Status: {warmup_status}")
            
            # Evaluation
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(eval_episodes)
                self.log(f"\n{'='*60}")
                self.log(f"Evaluation after {episode} episodes: {eval_reward:.3f}")
                self.log(f"{'='*60}\n")
            
            # Save checkpoint
            if episode % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_step{episode}.pt')
                self.save_checkpoint(checkpoint_path)
                self.log(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        self.log(f"\nTraining completed! Final model saved: {final_path}")
        
        # Save training log
        self.save_training_log(log_file)
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate agent với greedy policy - logic giống train_episode
        
        Args:
            num_episodes: Số episodes để evaluate
            
        Returns:
            avg_reward: Average reward
        """
        self.agent.eval()
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            
            action_buffer = len(angle_bins)
            angle_decision = None
            done = False
            reward_buffer = 0.0
            miss_streak = 0
            prev_num_items = -1
            prev_total_points = 0.0
            hidden_state = torch.zeros(1, self.agent.num_layers, self.agent.d_hidden).to(self.device) # batch first
            total_reward = 0.0
            
            while True:
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
                else:
                    if angle_decision is None or done:
                        env_f, item_f, mask = Embedder.preprocess_state(state)
                        cur_num_items = mask.sum().item()
                        
                        # Tính tổng point của các items hiện tại
                        cur_total_points = sum(item['point'] for item in state['items'])
                        
                        # Phát hiện TNT explosion: reward = 0 nhưng tổng point giảm
                        if reward_buffer == 0 and cur_total_points < prev_total_points:
                            lost_points = prev_total_points - cur_total_points
                            tnt_penalty = -self.env.c_tnt * lost_points / self.env.reward_scale
                            reward_buffer += tnt_penalty
                        
                        if cur_num_items == prev_num_items:
                            miss_streak += 1
                        else:
                            miss_streak = 0
                        prev_num_items = cur_num_items
                        prev_total_points = cur_total_points
                        
                        # Accumulate reward
                        total_reward += reward_buffer
                        
                        if done:
                            break
                        
                        # Greedy action selection
                        action_buffer, used_model, hidden_state = self.select_action(state, action_buffer, miss_streak, hidden_state, training=False)
                        angle_decision = angle_bins[action_buffer]
                        reward_buffer = 0
                    
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = self.env.step(1)  # Fire action
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
        
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
            
            eval_rewards.append(total_reward)
        
        self.agent.train()
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, path: str):
        """Lưu checkpoint"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'target_agent_state_dict': self.target_agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'replay_buffer': self.replay_buffer.buffer,  # Lưu replay buffer
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.target_agent.load_state_dict(checkpoint['target_agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        
        # Load replay buffer nếu có
        if 'replay_buffer' in checkpoint:
            # Load buffer và move tensors sang device
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.buffer = []
            for episode in buffer_data:
                # Move all tensors in episode to device
                episode_on_device = (
                    episode[0].to(self.device),  # env_feats
                    episode[1].to(self.device),  # item_feats
                    episode[2].to(self.device),  # masks
                    episode[3].to(self.device),  # actions
                    episode[4].to(self.device),  # rewards
                    episode[5]                    # seq_len (int, không cần move)
                )
                self.replay_buffer.buffer.append(episode_on_device)
            self.log(f"Loaded checkpoint from {path} (with {len(self.replay_buffer)} episodes in buffer, moved to {self.device})")
        else:
            self.log(f"Loaded checkpoint from {path} (no replay buffer)")

    
    def save_training_log(self, path: str):
        """Lưu training metrics"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'final_epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon_start': 0.5,  # Updated default
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'buffer_size': len(self.replay_buffer.buffer),
            }
        }
        
        with open(path, 'w') as f:
            json.dump(log, f, indent=2)
        self.log(f"Saved training log: {path}")