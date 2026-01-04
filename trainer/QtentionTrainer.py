"""
Deep Q-Learning Training cho Gold Miner
Sử dụng Qtention network với Experience Replay và Target Network
"""

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

from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Qtention import Qtention
from agent.Qtention.Embedder import Embedder

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

class ReplayBuffer:
    """
    Experience Replay Buffer để lưu transitions.
    
    Lưu preprocessed TORCH TENSORS (CPU) thay vì numpy arrays.
    Mỗi transition: (type_ids, item_feats, mov_idx, mov_feats, action, reward, 
                     next_type_ids, next_item_feats, next_mov_idx, next_mov_feats, done)
    
    sample() method TỰ ĐỘNG stack và pad các tensors thành batch ready-to-use.
    """
    
    def __init__(self, capacity: int = 5000):
        self.buffer: Deque = deque(maxlen=capacity)
    
    def push(self, type_ids: torch.Tensor, item_feats: torch.Tensor, mov_idx: torch.Tensor, mov_feats: torch.Tensor,
             action: int, reward: float, 
             next_type_ids: torch.Tensor, next_item_feats: torch.Tensor, next_mov_idx: torch.Tensor, next_mov_feats: torch.Tensor,
             done: bool):
        """
        Thêm transition vào buffer.
        Nhận torch tensors (CPU) - đã được preprocess sẵn.
        """
        # Đảm bảo mov tensors có shape đúng nếu empty
        mov_idx_t = mov_idx if len(mov_idx) > 0 else torch.empty(0, dtype=torch.long)
        mov_feats_t = mov_feats if len(mov_feats) > 0 else torch.empty(0, 3, dtype=torch.float32)
        
        next_mov_idx_t = next_mov_idx if len(next_mov_idx) > 0 else torch.empty(0, dtype=torch.long)
        next_mov_feats_t = next_mov_feats if len(next_mov_feats) > 0 else torch.empty(0, 3, dtype=torch.float32)
        
        self.buffer.append((
            type_ids, item_feats, mov_idx_t, mov_feats_t,
            action, reward,
            next_type_ids, next_item_feats, next_mov_idx_t, next_mov_feats_t,
            done
        ))
    
    def sample(self, batch_size: int) -> dict:
        """
        Sample random batch từ buffer và TỰ ĐỘNG stack + pad thành tensors.
        
        Returns:
            dict với keys (tất cả đều là torch.Tensor trên CPU):
                'type_ids': Tensor [B, max_L] - padded với PAD token (10)
                'item_feats': Tensor [B, max_L, 10] - padded với zeros
                'mov_idx': Tensor [B, max_mov_len] hoặc None
                'mov_feats': Tensor [B, max_mov_len, 3] hoặc None
                'actions': Tensor [B]
                'rewards': Tensor [B]
                'next_type_ids': Tensor [B, max_L]
                'next_item_feats': Tensor [B, max_L, 10]
                'next_mov_idx': Tensor [B, max_mov_len] hoặc None
                'next_mov_feats': Tensor [B, max_mov_len, 3] hoặc None
                'dones': Tensor [B]
        """
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        type_ids_list, item_feats_list, mov_idx_list, mov_feats_list, actions, rewards, \
        next_type_ids_list, next_item_feats_list, next_mov_idx_list, next_mov_feats_list, dones = zip(*batch)
        
        # Pad và stack type_ids và item_feats (VARIABLE LENGTH!)
        type_ids, item_feats = self._pad_item_features(type_ids_list, item_feats_list)
        next_type_ids, next_item_feats = self._pad_item_features(next_type_ids_list, next_item_feats_list)
        
        # Pad mov_idx và mov_feats (variable length)
        mov_idx, mov_feats = self._pad_movement_features(mov_idx_list, mov_feats_list)
        next_mov_idx, next_mov_feats = self._pad_movement_features(next_mov_idx_list, next_mov_feats_list)
        
        # Convert lists to tensors
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return {
            'type_ids': type_ids,
            'item_feats': item_feats,
            'mov_idx': mov_idx,
            'mov_feats': mov_feats,
            'actions': actions,
            'rewards': rewards,
            'next_type_ids': next_type_ids,
            'next_item_feats': next_item_feats,
            'next_mov_idx': next_mov_idx,
            'next_mov_feats': next_mov_feats,
            'dones': dones
        }
    
    def _pad_item_features(self, type_ids_list: List[torch.Tensor], item_feats_list: List[torch.Tensor]) -> Tuple:
        """
        Pad type_ids và item_feats to same length (max_L in batch).
        
        Returns:
            type_ids: Tensor [B, max_L] - padded với PAD token (10)
            item_feats: Tensor [B, max_L, d_feats] - padded với zeros
        """
        # Find max length in batch
        max_len = max(len(t) for t in type_ids_list)
        
        type_ids_padded = []
        item_feats_padded = []
        
        for type_ids, item_feats in zip(type_ids_list, item_feats_list):
            curr_len = len(type_ids)
            if curr_len < max_len:
                pad_len = max_len - curr_len
                # Pad type_ids với PAD token (10)
                type_ids_pad = torch.nn.functional.pad(type_ids, (0, pad_len), value=10)
                # Pad item_feats với zeros: (left, right, top, bottom, front, back)
                # item_feats shape: [L, d_feats] → pad last dimension: (0, 0, 0, pad_len)
                item_feats_pad = torch.nn.functional.pad(item_feats, (0, 0, 0, pad_len), value=0)
            else:
                type_ids_pad = type_ids
                item_feats_pad = item_feats
            
            type_ids_padded.append(type_ids_pad)
            item_feats_padded.append(item_feats_pad)
        
        return torch.stack(type_ids_padded), torch.stack(item_feats_padded)
    
    def _pad_movement_features(self, mov_idx_list: List[torch.Tensor], mov_feats_list: List[torch.Tensor]) -> Tuple:
        """
        Pad movement features to same length.
        
        Returns:
            mov_idx: Tensor [B, max_mov_len] hoặc None nếu không có movement
            mov_feats: Tensor [B, max_mov_len, 3] hoặc None nếu không có movement
        """
        max_mov_len = max(len(m) for m in mov_idx_list)
        
        if max_mov_len == 0:
            return None, None
        
        mov_idx_padded = []
        mov_feats_padded = []
        
        for m_idx, m_feat in zip(mov_idx_list, mov_feats_list):
            if len(m_idx) > 0:
                pad_len = max_mov_len - len(m_idx)
                # torch.nn.functional.pad: (left, right) cho 1D, (left, right, top, bottom) cho 2D
                m_idx_pad = torch.nn.functional.pad(m_idx, (0, pad_len), value=-1)
                m_feat_pad = torch.nn.functional.pad(m_feat, (0, 0, 0, pad_len), value=0)
            else:
                m_idx_pad = torch.full((max_mov_len,), -1, dtype=torch.long)
                m_feat_pad = torch.zeros(max_mov_len, 3, dtype=torch.float32)
            mov_idx_padded.append(m_idx_pad)
            mov_feats_padded.append(m_feat_pad)
        
        return torch.stack(mov_idx_padded), torch.stack(mov_feats_padded)
    
    def stack_batch(self, transitions: List[Tuple]) -> dict:
        """
        Stack một list transitions thành batch dict (dùng cho planning approach).
        
        Args:
            transitions: List of transitions từ buffer
            
        Returns:
            dict tương tự như sample() - tất cả đều torch.Tensor
        """
        # Unpack batch
        type_ids_list, item_feats_list, mov_idx_list, mov_feats_list, actions, rewards, \
        next_type_ids_list, next_item_feats_list, next_mov_idx_list, next_mov_feats_list, dones = zip(*transitions)
        
        # Pad và stack type_ids và item_feats (VARIABLE LENGTH!)
        type_ids, item_feats = self._pad_item_features(type_ids_list, item_feats_list)
        next_type_ids, next_item_feats = self._pad_item_features(next_type_ids_list, next_item_feats_list)
        
        # Pad mov_idx và mov_feats (variable length)
        mov_idx, mov_feats = self._pad_movement_features(mov_idx_list, mov_feats_list)
        next_mov_idx, next_mov_feats = self._pad_movement_features(next_mov_idx_list, next_mov_feats_list)
        
        # Convert lists to tensors
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return {
            'type_ids': type_ids,
            'item_feats': item_feats,
            'mov_idx': mov_idx,
            'mov_feats': mov_feats,
            'actions': actions,
            'rewards': rewards,
            'next_type_ids': next_type_ids,
            'next_item_feats': next_item_feats,
            'next_mov_idx': next_mov_idx,
            'next_mov_feats': next_mov_feats,
            'dones': dones
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class QtentionTrainer:
    """Trainer cho Deep Q-Learning"""
    
    def __init__(
        self,
        env: GoldMinerEnv,
        agent: Qtention,
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
        self.explore_strategy = explore_strategy
        self.num_episodes = num_episodes
        
        # Target network
        self.target_agent = Qtention(
            d_model=agent.d_model,
            n_actions=agent.n_actions,
            nhead=agent.nhead,
            n_layers=agent.n_layers,
            d_ff=agent.d_ff,
            dropout=agent.dropout,
            max_items=agent.embedder.max_items
        ).to(device)
        self.update_target_network()
        self.target_agent.eval()
        
        # Optimizer và loss
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tracking
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Logger (will be setup in train())
        self.logger = None
        
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
    
    def select_action(self, state: dict, miss_streak: int = 0, training: bool = True, k_random: int = 5) -> tuple:
        """
        Chọn action với epsilon-greedy policy
        
        Args:
            state: Game state dict
            miss_streak: Số lần miss liên tiếp
            training: Nếu True thì dùng epsilon-greedy, False thì greedy
            k_random: Số actions random để chọn max khi miss (default: 5)
            
        Returns:
            (action, used_model): action được chọn và flag cho biết có dùng model không
        """
        # Preprocess state
        type_ids, item_feats, mov_idx, mov_feats = Embedder.preprocess_state(state)
        
        # Convert to torch tensors và chuyển sang device
        type_ids_t = type_ids.unsqueeze(0).to(self.device)  # [1, L]
        item_feats_t = item_feats.unsqueeze(0).to(self.device)  # [1, L, d_feats]
        mov_idx_t = mov_idx.unsqueeze(0).to(self.device) if len(mov_idx) > 0 else None
        mov_feats_t = mov_feats.unsqueeze(0).to(self.device) if len(mov_feats) > 0 else None
        
        # Sau khi kéo hụt: random k actions và chọn max trong k đó (ưu tiên hơn epsilon-greedy)
        if miss_streak > 0:
            with torch.no_grad():
                q_values = self.agent(type_ids_t, item_feats_t, mov_idx_t, mov_feats_t)  # [1, n_actions]
                # Random k action indices
                random_actions = random.sample(range(self.agent.n_actions), min(k_random, self.agent.n_actions))
                # Lấy Q-values của các actions đó
                random_q_values = q_values[0, random_actions]
                # Chọn action có Q-value cao nhất trong k actions
                best_idx = random_q_values.argmax().item()
                action = random_actions[best_idx]
                q_value = random_q_values[best_idx].cpu().item()
            used_model = True
            action_mode = 'selective_random'
        # Epsilon-greedy action selection (chỉ khi miss_streak == 0)
        elif training and random.random() < self.epsilon:
            action = random.randint(0, self.agent.n_actions - 1)
            used_model = False
            q_value = None
            action_mode = 'random'
        else:
            with torch.no_grad():
                q_values = self.agent(type_ids_t, item_feats_t, mov_idx_t, mov_feats_t)  # [1, n_actions]
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0][action].cpu().item()
            used_model = True
            action_mode = 'model'
        
        # Lưu thông tin action vào global state để hiển thị trên màn hình
        from define import set_ai_action_info
        set_ai_action_info(action, q_value, used_model, action_mode)
        
        return action, used_model
    
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
        Planning approach: Quét qua toàn bộ buffer num_planning lần
        Mỗi planning step: quét qua toàn bộ buffer, mỗi phần tử một lần
        Batch cuối cùng có thể có size nhỏ hơn batch_size
        """
        losses = []
        
        # Quét qua buffer num_planning lần
        for planning_iter in range(self.num_planning):
            iter_loss = 0.0
            num_batches = 0
            
            # Lấy tất cả transitions từ buffer
            all_transitions = list(self.replay_buffer.buffer)
            total_samples = len(all_transitions)
            
            # Shuffle để tạo random order mỗi lần quét
            random.shuffle(all_transitions)
            
            # Quét qua toàn bộ buffer theo batches
            for batch_start in range(0, total_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_samples)
                transitions = all_transitions[batch_start:batch_end]
                
                # Stack transitions thành batch dict
                batch = self.replay_buffer.stack_batch(transitions)
                
                loss = self._train_on_batch(batch)
                iter_loss += loss
                num_batches += 1
            
            # Tính avg loss cho lần quét này
            avg_iter_loss = iter_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_iter_loss)
            
            # Log loss cho mỗi lần quét
                
            self.log(f"Step {cur_step}  Planning {planning_iter+1}/{self.num_planning} - Avg Loss: {avg_iter_loss:.8f} - Batches: {num_batches}/{(total_samples + self.batch_size - 1) // self.batch_size} - Buffer: {total_samples}")
        
        return losses
    
    def _train_step_standard(self, cur_step) -> list:
        """
        Standard DQN: Sample num_planning batches ngẫu nhiên và train
        """
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        losses = []
        
        # Sample và train trên num_planning batches
        for batch_idx in range(self.num_planning):
            # Sample random batch
            batch = self.replay_buffer.sample(self.batch_size)
            
            loss = self._train_on_batch(batch)
            losses.append(loss)
            
            # Log loss cho mỗi batch
            if cur_step % 6 == 0:
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
        # Move tensors from CPU to device (GPU/CPU) - KHÔNG CẦN torch.from_numpy nữa
        type_ids = batch['type_ids'].to(self.device)
        item_feats = batch['item_feats'].to(self.device)
        mov_idx = batch['mov_idx'].to(self.device) if batch['mov_idx'] is not None else None
        mov_feats = batch['mov_feats'].to(self.device) if batch['mov_feats'] is not None else None
        
        next_type_ids = batch['next_type_ids'].to(self.device)
        next_item_feats = batch['next_item_feats'].to(self.device)
        next_mov_idx = batch['next_mov_idx'].to(self.device) if batch['next_mov_idx'] is not None else None
        next_mov_feats = batch['next_mov_feats'].to(self.device) if batch['next_mov_feats'] is not None else None
        
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute Q(s, a)
        q_values = self.agent(type_ids, item_feats, mov_idx, mov_feats)  # [B, n_actions]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
        
        # Compute target
        with torch.no_grad():
            next_q_values = self.target_agent(next_type_ids, next_item_feats, next_mov_idx, next_mov_feats)  # [B, n_actions]
            next_q_values = next_q_values.max(1)[0]  # [B]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = self.loss_fn(q_values, targets)
        
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
        episode_reward = 0.0
        episode_steps = 0
        current_reward = 0
        action_buffer = None
        state_buffer = None
        new_state_buffer = None
        reward_buffer = 0
        angle_decision = None
        done = False
        prev_total_points = 0.0  # Track tổng point để detect TNT
        miss_streak = 0  # Số lần miss liên tiếp
        prev_num_items = -1  # Số items trước đó để detect miss
        
        def update_replay_buffer():
            nonlocal action_buffer, reward_buffer, episode_reward, episode_steps, prev_total_points
            
            # Phát hiện TNT explosion: reward = 0 nhưng tổng point giảm
            cur_total_points = sum(item['point'] for item in state['items'])
            if reward_buffer == 0 and cur_total_points < prev_total_points:
                lost_points = prev_total_points - cur_total_points
                tnt_penalty = -self.env.c_tnt * lost_points / self.env.reward_scale
                reward_buffer += tnt_penalty
            prev_total_points = cur_total_points
            
            old_type_ids, old_item_feats, old_mov_idx, old_mov_feats = state_buffer
            new_type_ids, new_item_feats, new_mov_idx, new_mov_feats = Embedder.preprocess_state(state)
            self.replay_buffer.push(
                old_type_ids, old_item_feats, old_mov_idx, old_mov_feats,
                action_buffer, reward_buffer,
                new_type_ids, new_item_feats, new_mov_idx, new_mov_feats,
                done
            )
            
            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()
            if self.total_steps % self.train_freq == 0:
                self.log(f"Episode step {episode_steps} reward: {episode_reward}")
                losses = self.train_step(self.total_steps)
                if losses:  # Nếu có loss (buffer đủ lớn)
                    self.losses.extend(losses)  # Thêm tất cả losses vào list
                    
            episode_reward += reward_buffer
            episode_steps += 1
            
            reward_buffer = 0
            action_buffer = None
            
            return new_type_ids, new_item_feats, new_mov_idx, new_mov_feats
        
        while True:
            if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
            else:
                if angle_decision is None or done:
                    if state_buffer is not None:     
                        new_state_buffer = update_replay_buffer()
                        
                        # Track miss_streak
                        cur_num_items = len(state['items'])
                        if cur_num_items == prev_num_items:
                            miss_streak += 1
                        else:
                            miss_streak = 0
                        prev_num_items = cur_num_items
                    if done:
                        break
                    action_buffer, used_model = self.select_action(state, miss_streak=miss_streak, training=True)
                    angle_decision = angle_bins[action_buffer]
                    state_buffer = new_state_buffer if state_buffer is not None else Embedder.preprocess_state(state)
                
                current_angle = state['rope_state']['direction']
                if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                    next_state, reward, terminated, truncated, info = self.env.step(1)  # Fire action
                    angle_decision = None
                else:
                    next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
    
            done = terminated or truncated
            reward_buffer += reward
            state = next_state
        
        # Decay epsilon (sử dụng self.num_episodes từ train loop)
        if hasattr(self, '_current_episode') and self.explore_strategy == 'Linearly':
            self.epsilon = max(self.epsilon_end, 
                              self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self._current_episode / self.num_episodes))
        else:  # Exponentially (default)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return episode_reward, episode_steps
    
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
        self.log(f"Replay buffer size: {len(self.replay_buffer)}")
        self.log("-" * 60)
        
        for episode in range(1, num_episodes + 1):
            self.log(f"\nStarting Episode {episode}/{num_episodes}")
            
            # Set current episode for linear decay
            self._current_episode = episode
            
            # Train episode
            episode_reward, episode_steps = self.train_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # In điểm kiếm được sau mỗi episode
            self.log(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps}")
            print(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps}")
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0.0
                
                self.log(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.3f} | "
                      f"Avg(10): {avg_reward:.8f} | "
                      f"Steps: {episode_steps} | "
                      f"Loss: {avg_loss:.8f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer)}")
            
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
            episode_reward = 0.0
            episode_steps = 0
            action_buffer = None
            reward_buffer = 0
            angle_decision = None
            done = False
            prev_total_points = 0.0  # Track tổng point để detect TNT
            miss_streak = 0  # Số lần miss liên tiếp
            prev_num_items = -1  # Số items trước đó để detect miss
            
            while True:
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
                else:
                    if angle_decision is None or done:
                        if action_buffer is not None:
                            # Phát hiện TNT explosion
                            cur_total_points = sum(item['point'] for item in state['items'])
                            if reward_buffer == 0 and cur_total_points < prev_total_points:
                                lost_points = prev_total_points - cur_total_points
                                tnt_penalty = -self.env.c_tnt * lost_points / self.env.reward_scale
                                reward_buffer += tnt_penalty
                            prev_total_points = cur_total_points
                            
                            # Track miss_streak
                            cur_num_items = len(state['items'])
                            if cur_num_items == prev_num_items:
                                miss_streak += 1
                            else:
                                miss_streak = 0
                            prev_num_items = cur_num_items
                            
                            episode_reward += reward_buffer
                            episode_steps += 1
                            reward_buffer = 0
                            action_buffer = None
                        if done:
                            break
                        action_buffer, used_model = self.select_action(state, miss_streak=miss_streak, training=False)  # Greedy
                        angle_decision = angle_bins[action_buffer]
                    
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = self.env.step(1)  # Fire action
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
        
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
        
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
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        
        # Load replay buffer nếu có
        if 'replay_buffer' in checkpoint:
            # Load buffer và move tensors sang device
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.buffer = []
            for transition in buffer_data:
                # Move all tensors in transition to device
                # transition = (env_feats, item_feats, mask, action, reward, next_env_feats, next_item_feats, next_mask, done)
                transition_on_device = (
                    transition[0].to(self.device),  # env_feats
                    transition[1].to(self.device),  # item_feats
                    transition[2].to(self.device),  # mask
                    transition[3],                   # action (int)
                    transition[4],                   # reward (float)
                    transition[5].to(self.device),  # next_env_feats
                    transition[6].to(self.device),  # next_item_feats
                    transition[7].to(self.device),  # next_mask
                    transition[8]                    # done (bool)
                )
                self.replay_buffer.buffer.append(transition_on_device)
            self.log(f"Loaded checkpoint from {path} (with {len(self.replay_buffer)} transitions in buffer, moved to {self.device})")
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
                'buffer_size': self.replay_buffer.buffer.maxlen,
            }
        }
        
        with open(path, 'w') as f:
            json.dump(log, f, indent=2)
        self.log(f"Saved training log: {path}")