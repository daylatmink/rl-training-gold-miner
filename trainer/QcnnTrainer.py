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
from agent.QCNN.QCNN import QCNN
from agent.QCNN.Embedder import Embedder

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
    Experience Replay Buffer cho QCNN.
    
    Lưu preprocessed TORCH TENSORS với padding cứng về max_items.
    Mỗi transition: (env_feats, item_feats, mask, action, reward, 
                     next_env_feats, next_item_feats, next_mask, done)
    """
    
    def __init__(self, capacity: int = 5000, device: str = 'cpu'):
        """
        Args:
            capacity: Total buffer capacity
            device: Device to store tensors
        """
        self.buffer: Deque = deque(maxlen=capacity)
        self.device = device
    
    def push(self, env_feats: torch.Tensor, item_feats: torch.Tensor, mask: torch.Tensor,
             action: int, reward: float, 
             next_env_feats: torch.Tensor, next_item_feats: torch.Tensor, next_mask: torch.Tensor,
             done: bool):
        """
        Thêm transition vào buffer.
        
        Args:
            env_feats: [10] - Environment features
            item_feats: [max_items, 23] - Item features (padded)
            mask: [max_items] - Mask (1=real, 0=padding)
            action: int - Action taken
            reward: float - Reward received
            next_env_feats: [10] - Next environment features
            next_item_feats: [max_items, 23] - Next item features (padded)
            next_mask: [max_items] - Next mask
            done: bool - Episode done flag
        """
        # Move tensors to device when pushing
        self.buffer.append((
            env_feats.to(self.device), 
            item_feats.to(self.device), 
            mask.to(self.device),
            action, reward,
            next_env_feats.to(self.device), 
            next_item_feats.to(self.device), 
            next_mask.to(self.device),
            done
        ))
    
    def sample(self, batch_size: int) -> dict:
        """
        Sample random batch từ buffer.
        
        Args:
            batch_size: Batch size
            
        Returns:
            dict với keys (tất cả đều là torch.Tensor trên CPU):
                'env_feats': Tensor [B, 10] - Environment features
                'item_feats': Tensor [B, max_items, 23] - Item features (padded)
                'mask': Tensor [B, max_items] - Mask (1=real, 0=padding)
                'actions': Tensor [B]
                'rewards': Tensor [B]
                'next_env_feats': Tensor [B, 10]
                'next_item_feats': Tensor [B, max_items, 23] - Next item features (padded)
                'next_mask': Tensor [B, max_items] - Next mask
                'dones': Tensor [B]
        """
        # Sample random transitions từ buffer
        transitions = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        env_feats_list, item_feats_list, mask_list, actions, rewards, \
        next_env_feats_list, next_item_feats_list, next_mask_list, dones = zip(*transitions)
        
        # Stack tensors (tất cả đã có cùng shape do padding)
        env_feats = torch.stack(env_feats_list)  # [B, 10]
        item_feats = torch.stack(item_feats_list)  # [B, max_items, 23]
        mask = torch.stack(mask_list)  # [B, max_items]
        next_env_feats = torch.stack(next_env_feats_list)  # [B, 10]
        next_item_feats = torch.stack(next_item_feats_list)  # [B, max_items, 23]
        next_mask = torch.stack(next_mask_list)  # [B, max_items]
        
        # Convert lists to tensors
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return {
            'env_feats': env_feats,
            'item_feats': item_feats,
            'mask': mask,
            'actions': actions,
            'rewards': rewards,
            'next_env_feats': next_env_feats,
            'next_item_feats': next_item_feats,
            'next_mask': next_mask,
            'dones': dones
        }
    
    def __len__(self) -> int:
        """Tổng số transitions trong buffer"""
        return len(self.buffer)


class QcnnTrainer:
    """Trainer cho Deep Q-Learning"""
    
    def __init__(
        self,
        env: GoldMinerEnv,
        agent: QCNN,
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
        self.target_agent = QCNN(
            d_model=agent.d_model,
            n_actions=agent.n_actions,
            d_hidden=agent.d_hidden,
        ).to(device)
        self.update_target_network()
        self.target_agent.eval()
        
        # Optimizer và loss
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
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
            
            for transition in buffer_data:
                self.replay_buffer.buffer.append(transition)
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
    
    def select_action(self, state: dict, training: bool = True) -> tuple:
        """
        Chọn action với epsilon-greedy policy
        
        Args:
            state: Game state dict
            training: Nếu True thì dùng epsilon-greedy, False thì greedy
            
        Returns:
            (action, used_model): action được chọn và flag cho biết có dùng model không
        """
        # Preprocess state for QCNN
        env_feats, item_feats, mask = Embedder.preprocess_state(state, return_batch=True)
        
        # Convert to device
        env_feats = env_feats.to(self.device)  # [1, 10]
        item_feats = item_feats.to(self.device)  # [1, max_items, 23]
        mask = mask.to(self.device)  # [1, max_items]
        
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
                q_values = self.agent(env_feats, item_feats, mask)  # [1, n_actions]
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0][action].cpu().item()
            used_model = True
        
        # Lưu thông tin action vào global state để hiển thị trên màn hình
        from define import set_ai_action_info
        set_ai_action_info(action, q_value, used_model)
        
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
            all_transitions = list(self.replay_buffer.buffer)
            random.shuffle(all_transitions)
            
            # Quét qua toàn bộ buffer theo batches
            for batch_start in range(0, len(all_transitions), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(all_transitions))
                
                if batch_end - batch_start < self.batch_size:
                    continue  # Skip batch cuối nếu không đủ size
                
                # Lấy batch và unpack
                batch_transitions = all_transitions[batch_start:batch_end]
                env_feats_list, item_feats_list, mask_list, actions, rewards, \
                next_env_feats_list, next_item_feats_list, next_mask_list, dones = zip(*batch_transitions)
                
                # Stack thành batch dict
                batch = {
                    'env_feats': torch.stack(env_feats_list),
                    'item_feats': torch.stack(item_feats_list),
                    'mask': torch.stack(mask_list),
                    'actions': torch.tensor(actions, dtype=torch.long),
                    'rewards': torch.tensor(rewards, dtype=torch.float32),
                    'next_env_feats': torch.stack(next_env_feats_list),
                    'next_item_feats': torch.stack(next_item_feats_list),
                    'next_mask': torch.stack(next_mask_list),
                    'dones': torch.tensor(dones, dtype=torch.float32)
                }
                
                loss = self._train_on_batch(batch)
                iter_loss += loss
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
        # Move tensors from CPU to device (GPU/CPU)
        env_feats = batch['env_feats'].to(self.device)  # [B, 10]
        item_feats = batch['item_feats'].to(self.device)  # [B, max_items, 23]
        mask = batch['mask'].to(self.device)  # [B, max_items]
        next_env_feats = batch['next_env_feats'].to(self.device)  # [B, 10]
        next_item_feats = batch['next_item_feats'].to(self.device)  # [B, max_items, 23]
        next_mask = batch['next_mask'].to(self.device)  # [B, max_items]
        
        actions = batch['actions'].to(self.device)  # [B]
        rewards = batch['rewards'].to(self.device)  # [B]
        dones = batch['dones'].to(self.device)  # [B]
        
        # Compute Q(s, a) for QCNN
        q_values = self.agent(env_feats, item_feats, mask)  # [B, n_actions]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
        
        # Compute target
        with torch.no_grad():
            next_q_values = self.target_agent(next_env_feats, next_item_feats, next_mask)  # [B, n_actions]
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
        
        def update_replay_buffer():
            nonlocal action_buffer, reward_buffer, episode_reward, episode_steps, prev_total_points
            
            # Phát hiện TNT explosion: reward = 0 nhưng tổng point giảm
            cur_total_points = sum(item['point'] for item in state['items'])
            if reward_buffer == 0 and cur_total_points < prev_total_points:
                lost_points = prev_total_points - cur_total_points
                tnt_penalty = -self.env.c_tnt * lost_points / self.env.reward_scale
                reward_buffer += tnt_penalty
            prev_total_points = cur_total_points
            
            old_env_feats, old_item_feats, old_mask = state_buffer
            new_env_feats, new_item_feats, new_mask = Embedder.preprocess_state(state)
            self.replay_buffer.push(
                old_env_feats, old_item_feats, old_mask,
                action_buffer, reward_buffer,
                new_env_feats, new_item_feats, new_mask,
                done
            )
            
            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()
            # Chỉ train sau khi warmup xong
            if len(self.replay_buffer) >= self.warmup_steps and self.total_steps % self.train_freq == 0:
                self.log(f"Episode step {episode_steps} reward: {episode_reward}")
                losses = self.train_step(self.total_steps)
                if losses:  # Nếu có loss (buffer đủ lớn)
                    self.losses.extend(losses)  # Thêm tất cả losses vào list
                    
            episode_reward += reward_buffer
            episode_steps += 1
            
            reward_buffer = 0
            action_buffer = None
            
            return new_env_feats, new_item_feats, new_mask
        
        while True:
            if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                next_state, reward, terminated, truncated, info = self.env.step(0)  # No-op action
            else:
                if angle_decision is None or done:
                    if state_buffer is not None:     
                        new_state_buffer = update_replay_buffer()
                    if done:
                        break
                    action_buffer, used_model = self.select_action(state, training=True)
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
        self.log(f"Warmup steps: {self.warmup_steps} (random actions only)")
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
            buffer_size = len(self.replay_buffer)
            self.log(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps} | Buffer: {buffer_size}")
            print(f"Episode {episode} completed | Score: {episode_reward:.3f} | Steps: {episode_steps} | Buffer: {buffer_size}")
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0.0
                warmup_status = "WARMUP" if self.total_steps < self.warmup_steps else "TRAINING"
                
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
            episode_reward = 0.0
            episode_steps = 0
            action_buffer = None
            reward_buffer = 0
            angle_decision = None
            done = False
            prev_total_points = 0.0  # Track tổng point để detect TNT
            
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
                            
                            episode_reward += reward_buffer
                            episode_steps += 1
                            reward_buffer = 0
                            action_buffer = None
                        if done:
                            break
                        action_buffer, used_model = self.select_action(state, training=False)  # Greedy
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
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        
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

    def analyze_action_distribution(self, num_samples: int = 1000):
        """
        Debug helper: kiểm tra mạng có thực sự phân biệt state hay không
        bằng cách thống kê phân bố argmax(Q(s, a)) trên một tập state
        lấy từ replay buffer.

        Args:
            num_samples: số lượng transition tối đa lấy ra để phân tích
        """
        if len(self.replay_buffer) == 0:
            print("Replay buffer is empty, nothing to analyze.")
            return

        # Lấy ngẫu nhiên tối đa num_samples transition từ buffer
        buffer_list = list(self.replay_buffer.buffer)
        n = min(num_samples, len(buffer_list))
        transitions = random.sample(buffer_list, n)

        # Unpack các tensor state từ transition
        env_feats_list, item_feats_list, mask_list, actions, rewards, \
            next_env_feats_list, next_item_feats_list, next_mask_list, dones = zip(*transitions)

        env_feats = torch.stack(env_feats_list).to(self.device)          # [N, 10]
        item_feats = torch.stack(item_feats_list).to(self.device)        # [N, max_items, 23]
        mask = torch.stack(mask_list).to(self.device)                    # [N, max_items]

        # Chạy forward qua mạng và lấy argmax
        self.agent.eval()
        with torch.no_grad():
            q_values = self.agent(env_feats, item_feats, mask)           # [N, n_actions]
            argmax_actions = q_values.argmax(dim=1).cpu().numpy()        # [N]

        # Thống kê phân bố action
        unique_actions, counts = np.unique(argmax_actions, return_counts=True)
        print(f"\n[Analyze] Argmax(Q(s,a)) distribution over {n} sampled states:")
        for a, c in zip(unique_actions, counts):
            frac = c / n
            print(f"  Action {int(a):2d}: {c:4d} samples ({frac:6.2%})")

        # Một số thống kê phụ trợ
        avg_q = q_values.mean().item()
        max_q = q_values.max().item()
        min_q = q_values.min().item()
        print(f"[Analyze] Q statistics: avg={avg_q:.4f}, min={min_q:.4f}, max={max_q:.4f}")
        print(f"[Analyze] Number of distinct argmax actions: {len(unique_actions)}/{self.agent.n_actions}")

        # Trả agent về mode train như cũ
        self.agent.train()
