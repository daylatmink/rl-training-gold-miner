"""
Double Deep Q-Learning Training cho Gold Miner với Qtention Network
Strategy:
1. Load warmup buffer
2. Q-step update n lần
3. Chạy m episodes và lưu vào buffer
4. Lặp lại
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
import pickle
from datetime import datetime
from tqdm import tqdm
import logging
import torch.nn.functional as F

from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Qtention import Qtention
from agent.Qtention.Embedder import Embedder
from define import set_ai_action_info

# 50 actions - chia đều góc 15-165° thành 50 khoảng (3° mỗi khoảng)
angle_bins = [
    (15.0, 18.0), (18.0, 21.0), (21.0, 24.0), (24.0, 27.0), (27.0, 30.0),
    (30.0, 33.0), (33.0, 36.0), (36.0, 39.0), (39.0, 42.0), (42.0, 45.0),
    (45.0, 48.0), (48.0, 51.0), (51.0, 54.0), (54.0, 57.0), (57.0, 60.0),
    (60.0, 63.0), (63.0, 66.0), (66.0, 69.0), (69.0, 72.0), (72.0, 75.0),
    (75.0, 78.0), (78.0, 81.0), (81.0, 84.0), (84.0, 87.0), (87.0, 90.0),
    (90.0, 93.0), (93.0, 96.0), (96.0, 99.0), (99.0, 102.0), (102.0, 105.0),
    (105.0, 108.0), (108.0, 111.0), (111.0, 114.0), (114.0, 117.0), (117.0, 120.0),
    (120.0, 123.0), (123.0, 126.0), (126.0, 129.0), (129.0, 132.0), (132.0, 135.0),
    (135.0, 138.0), (138.0, 141.0), (141.0, 144.0), (144.0, 147.0), (147.0, 150.0),
    (150.0, 153.0), (153.0, 156.0), (156.0, 159.0), (159.0, 162.0), (162.0, 165.0)
]


class ReplayBuffer:
    """
    Replay Buffer cho Qtention với padded transitions.
    
    Mỗi transition: (type_ids, item_feats, mov_idx, mov_feats, actual_length,
                     action, reward, 
                     next_type_ids, next_item_feats, next_mov_idx, next_mov_feats, next_actual_length,
                     done)
    
    All tensors are already PADDED to max_items + 1.
    """
    
    def __init__(self, capacity: int = 5000, device='cpu', n_actions: int = 50):
        self.buffer: Deque = deque(maxlen=capacity)
        self.device = device
        self.n_actions = n_actions
        # Track action counts cho inverse frequency exploration
        self.action_counts = np.zeros(n_actions, dtype=np.float32)
    
    def push(self, type_ids, item_feats, mov_idx, mov_feats, actual_length,
             action, reward,
             next_type_ids, next_item_feats, next_mov_idx, next_mov_feats, next_actual_length,
             done):
        """
        Thêm transition vào buffer. Tensors đã được pad sẵn từ warmup script.
        """
        # Nếu buffer đầy, giảm count của action bị xóa
        if len(self.buffer) == self.buffer.maxlen:
            old_action = self.buffer[0][5]  # action ở index 5
            self.action_counts[old_action] = max(0, self.action_counts[old_action] - 1)
        
        # Thêm transition mới
        self.buffer.append((
            type_ids.to(self.device), item_feats.to(self.device), 
            mov_idx.to(self.device), mov_feats.to(self.device), actual_length,
            action, reward,
            next_type_ids.to(self.device), next_item_feats.to(self.device),
            next_mov_idx.to(self.device), next_mov_feats.to(self.device), next_actual_length,
            done
        ))
        
        # Update action count
        self.action_counts[action] += 1
    
    def sample(self, batch_size: int) -> dict:
        """
        Sample random batch. Vì đã pad sẵn, chỉ cần stack.
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack
        type_ids_list, item_feats_list, mov_idx_list, mov_feats_list, lengths, \
        actions, rewards, \
        next_type_ids_list, next_item_feats_list, next_mov_idx_list, next_mov_feats_list, next_lengths, \
        dones = zip(*batch)
        
        # Stack tensors (already same shape due to padding)
        return {
            'type_ids': torch.stack(type_ids_list),
            'item_feats': torch.stack(item_feats_list),
            'mov_idx': self._stack_1d_with_padding(mov_idx_list, pad_value=-1),
            'mov_feats': self._stack_2d_with_padding(mov_feats_list, feat_dim=3, pad_value=0.0),
            'lengths': torch.tensor(lengths, dtype=torch.long, device=self.device),
            'actions': torch.tensor(actions, dtype=torch.long, device=self.device),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=self.device),
            'next_type_ids': torch.stack(next_type_ids_list),
            'next_item_feats': torch.stack(next_item_feats_list),
            'next_mov_idx': self._stack_1d_with_padding(next_mov_idx_list, pad_value=-1),
            'next_mov_feats': self._stack_2d_with_padding(next_mov_feats_list, feat_dim=3, pad_value=0.0),
            'next_lengths': torch.tensor(next_lengths, dtype=torch.long, device=self.device),
            'dones': torch.tensor(dones, dtype=torch.float32, device=self.device)
        }
    
    def _stack_1d_with_padding(self, tensor_list, pad_value=-1):
        """Stack 1D tensors (mov_idx) với padding"""
        sizes = [t.size(0) for t in tensor_list]
        max_len = max(sizes)
        
        if max_len == 0:
            return None
        
        padded = []
        for t, size in zip(tensor_list, sizes):
            if size == 0:
                t = torch.full((max_len,), pad_value, dtype=torch.long, device=self.device)
            elif size < max_len:
                t = F.pad(t, (0, max_len - size), value=pad_value)
            padded.append(t)
        
        return torch.stack(padded)
    
    def _stack_2d_with_padding(self, tensor_list, feat_dim, pad_value=0.0):
        """Stack 2D tensors (mov_feats) với padding"""
        sizes = [t.size(0) for t in tensor_list]
        max_len = max(sizes)
        
        if max_len == 0:
            return None
        
        padded = []
        for t, size in zip(tensor_list, sizes):
            if size == 0:
                t = torch.full((max_len, feat_dim), pad_value, dtype=torch.float32, device=self.device)
            elif size < max_len:
                t = F.pad(t, (0, 0, 0, max_len - size), value=pad_value)
            padded.append(t)
        
        return torch.stack(padded)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_explore_probs(self, temperature: float = 0.5) -> np.ndarray:
        """
        Tính xác suất explore dựa trên inverse frequency.
        Actions xuất hiện ít sẽ có xác suất cao hơn.
        
        Args:
            temperature: Softmax temperature (cao hơn = phẳng hơn)
        
        Returns:
            probs: [n_actions] xác suất cho mỗi action
        """
        # Inverse frequency: ít xuất hiện -> weight cao
        # Thêm 1 để tránh division by zero
        inverse_freq = 1.0 / (self.action_counts + 1)
        
        # Softmax với temperature
        exp_weights = np.exp(inverse_freq / temperature)
        probs = exp_weights / exp_weights.sum()
        
        return probs
    
    def sample_action_by_inverse_freq(self, temperature: float = 1.0) -> int:
        """
        Sample action theo inverse frequency.
        
        Returns:
            action: action index
        """
        probs = self.get_explore_probs(temperature)
        return np.random.choice(self.n_actions, p=probs)


class DoubleQtentionTrainer:
    """
    Double DQN Trainer với strategy:
    1. Load buffer
    2. Update n lần
    3. Collect m episodes
    4. Repeat
    """
    
    def __init__(
        self,
        env: GoldMinerEnv,
        agent: Qtention,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        max_items: int = 30,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.agent = agent.to(device)
        self.device = device
        self.max_items = max_items
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Target network
        self.target_agent = Qtention(
            d_model=agent.d_model,
            d_ff=agent.d_ff,
            nhead=agent.nhead,
            n_layers=agent.n_layers,
            dropout=agent.dropout,
            max_items=max_items,
            n_actions=agent.n_actions
        ).to(device)
        self.update_target_network()
        self.target_agent.eval()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr, weight_decay=1e-5)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, device=device, n_actions=agent.n_actions)
        
        # Exploration temperature cho inverse frequency sampling
        self.explore_temperature = 0.5
        
        # Tracking
        self.total_steps = 0
        self.episode_rewards = []
        self.losses = []
        
        # Logger
        self.logger = None
    
    def load_warmup_buffer(self, warmup_path: str):
        """Load warmup buffer từ file"""
        print(f"\n[1/4] Loading warmup buffer from {warmup_path}...")
        with open(warmup_path, 'rb') as f:
            buffer_data = pickle.load(f)
        
        print(f"  Loading {len(buffer_data)} transitions...")
        for transition in tqdm(buffer_data, desc="Loading buffer"):
            self.replay_buffer.push(*transition)
        
        print(f"✓ Buffer loaded: {len(self.replay_buffer)} transitions")
        
        # In action distribution
        counts = self.replay_buffer.action_counts
        print(f"\n  Action distribution in buffer:")
        print(f"    Min count: {counts.min():.0f}, Max count: {counts.max():.0f}, Mean: {counts.mean():.1f}")
        zero_actions = (counts == 0).sum()
        if zero_actions > 0:
            print(f"    Warning: {zero_actions} actions have 0 samples!")
    
    def setup_logger(self, log_file: str = 'training.log'):
        """Setup logger"""
        self.logger = logging.getLogger('DoubleQtentionTrainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def log(self, message: str):
        """Log message"""
        if self.logger:
            self.logger.info(message)
        print(message)
    
    def update_target_network(self):
        """Copy weights từ online agent sang target agent"""
        self.target_agent.load_state_dict(self.agent.state_dict())
    
    def create_mask(self, type_ids, actual_length):
        """
        Tạo attention mask từ actual_length.
        
        Args:
            type_ids: [B, L] - không dùng nhưng để biết L
            actual_length: [B] - số tokens thực tế
        
        Returns:
            mask: [B, L] - True cho padding positions
        """
        B, L = type_ids.shape
        # Create position indices: [B, L]
        positions = torch.arange(L, device=type_ids.device).unsqueeze(0).expand(B, -1)
        # Mask: True where position >= actual_length
        mask = positions >= actual_length.unsqueeze(1)
        return mask
    
    @torch.no_grad()
    def select_action(self, state: dict, training: bool = True) -> Tuple[int, bool]:
        """
        Epsilon-greedy action selection với inverse frequency exploration.
        Khi explore, ưu tiên các action ít xuất hiện trong buffer.
        
        Returns:
            (action, used_model): action index and whether model was used
        """
        q_value = None  # Initialize
        
        if training and random.random() < self.epsilon:
            # Explore: sample theo inverse frequency của buffer
            action = self.replay_buffer.sample_action_by_inverse_freq(self.explore_temperature)
            used_model = False
        else:
            # Preprocess state
            type_ids, item_feats, mov_idx, mov_feats = Embedder.preprocess_state(state, max_items=self.max_items)
            
            # Pad to max_length
            max_length = self.max_items + 1
            actual_length = len(type_ids)
            
            if actual_length < max_length:
                pad_size = max_length - actual_length
                type_ids = torch.cat([type_ids, torch.full((pad_size,), Embedder.TOKEN_TYPES['PAD'], dtype=torch.int64)])
                item_feats = torch.cat([item_feats, torch.zeros((pad_size, 10), dtype=torch.float32)])
            
            # Create mask
            mask = torch.arange(max_length) >= actual_length
            
            # Add batch dimension and move to device
            type_ids = type_ids.unsqueeze(0).to(self.device)
            item_feats = item_feats.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            
            if mov_idx.numel() > 0:
                mov_idx = mov_idx.unsqueeze(0).to(self.device)
                mov_feats = mov_feats.unsqueeze(0).to(self.device)
            else:
                mov_idx = None
                mov_feats = None
            
            # Forward pass
            q_values = self.agent(type_ids, item_feats, mov_idx, mov_feats, mask)
            action = q_values.argmax(dim=1).item()
            q_value = q_values[0, action].item()
            used_model = True
        
        # Set AI action info for display
        set_ai_action_info(action, q_value, used_model)
        
        return action, used_model
    
    def train_step(self, n_updates: int) -> List[float]:
        """
        Update agent n lần bằng Double DQN.
        
        Args:
            n_updates: Số lần update
        
        Returns:
            losses: List of losses
        """
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        losses = []
        
        for _ in range(n_updates):
            # Sample batch
            batch = self.replay_buffer.sample(self.batch_size)
            
            # Create masks
            mask = self.create_mask(batch['type_ids'], batch['lengths'])
            next_mask = self.create_mask(batch['next_type_ids'], batch['next_lengths'])
            
            # Forward pass - Online network
            q_values = self.agent(
                batch['type_ids'], batch['item_feats'],
                batch['mov_idx'], batch['mov_feats'], mask
            )  # [B, n_actions]
            
            # Q(s, a)
            q_sa = q_values.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)  # [B]
            
            # Double DQN target
            with torch.no_grad():
                # Online network chọn action
                next_q_online = self.agent(
                    batch['next_type_ids'], batch['next_item_feats'],
                    batch['next_mov_idx'], batch['next_mov_feats'], next_mask
                )  # [B, n_actions]
                best_next_actions = next_q_online.argmax(dim=1)  # [B]
                
                # Target network đánh giá action
                next_q_target = self.target_agent(
                    batch['next_type_ids'], batch['next_item_feats'],
                    batch['next_mov_idx'], batch['next_mov_feats'], next_mask
                )  # [B, n_actions]
                next_q = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)  # [B]
                
                # Target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
                target = batch['rewards'] + self.gamma * next_q * (1 - batch['dones'])
            
            # Loss
            loss = F.smooth_l1_loss(q_sa, target)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            
            # Update target network
            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()
        
        return losses
    
    def collect_episodes(self, m_episodes: int) -> List[Tuple[float, int]]:
        """
        Collect m episodes và lưu vào buffer.
        
        Returns:
            List of (total_reward, steps) for each episode
        """
        episode_results = []
        
        for _ in range(m_episodes):
            state, _ = self.env.reset()
            
            action_buffer = None
            state_buffer = None
            reward_buffer = 0.0
            angle_decision = None
            done = False
            episode_steps = 0
            episode_reward = 0.0
            
            while True:
                # Wait for rope animation
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or 
                                state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = self.env.step(0)
                else:
                    if angle_decision is None or done:
                        if state_buffer is not None:
                            # Save transition
                            old_type_ids, old_item_feats, old_mov_idx, old_mov_feats, old_length = state_buffer
                            
                            # Preprocess new state
                            new_type_ids, new_item_feats, new_mov_idx, new_mov_feats = Embedder.preprocess_state(
                                state, max_items=self.max_items
                            )
                            
                            # Pad new state
                            max_length = self.max_items + 1
                            new_length = len(new_type_ids)
                            if new_length < max_length:
                                pad_size = max_length - new_length
                                new_type_ids = torch.cat([new_type_ids, torch.full((pad_size,), Embedder.TOKEN_TYPES['PAD'], dtype=torch.int64)])
                                new_item_feats = torch.cat([new_item_feats, torch.zeros((pad_size, 10), dtype=torch.float32)])
                            
                            # Push to buffer
                            self.replay_buffer.push(
                                old_type_ids, old_item_feats, old_mov_idx, old_mov_feats, old_length,
                                action_buffer, reward_buffer,
                                new_type_ids, new_item_feats, new_mov_idx, new_mov_feats, new_length,
                                done
                            )
                            
                            episode_steps += 1
                            episode_reward += reward_buffer
                        
                        if done:
                            break
                        
                        # Select action
                        action_buffer, used_model = self.select_action(state, training=True)
                        angle_decision = angle_bins[action_buffer]
                        
                        # Preprocess and pad state
                        type_ids, item_feats, mov_idx, mov_feats = Embedder.preprocess_state(state, max_items=self.max_items)
                        max_length = self.max_items + 1
                        actual_length = len(type_ids)
                        
                        if actual_length < max_length:
                            pad_size = max_length - actual_length
                            type_ids = torch.cat([type_ids, torch.full((pad_size,), Embedder.TOKEN_TYPES['PAD'], dtype=torch.int64)])
                            item_feats = torch.cat([item_feats, torch.zeros((pad_size, 10), dtype=torch.float32)])
                        
                        state_buffer = (type_ids, item_feats, mov_idx, mov_feats, actual_length)
                        reward_buffer = 0.0
                    
                    # Execute action
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = self.env.step(1)
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = self.env.step(0)
                
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
            
            episode_results.append((episode_reward, episode_steps))
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return episode_results
    
    def train(self, n_cycles: int, n_updates_per_cycle: int, m_episodes_per_cycle: int, 
              warmup_buffer_path: str = None, save_dir: str = 'checkpoints', 
              log_file: str = 'training.log'):
        """
        Main training loop.
        
        Args:
            n_cycles: Số cycles
            n_updates_per_cycle: Số updates mỗi cycle
            m_episodes_per_cycle: Số episodes collect mỗi cycle
            warmup_buffer_path: Path to warmup buffer (optional)
            save_dir: Directory to save checkpoints
            log_file: Log file path
        """
        os.makedirs(save_dir, exist_ok=True)
        self.setup_logger(log_file)
        
        # Load warmup buffer if provided
        if warmup_buffer_path:
            self.load_warmup_buffer(warmup_buffer_path)
        
        self.log("\n" + "="*70)
        self.log("Double Qtention Training Started")
        self.log("="*70)
        self.log(f"Cycles: {n_cycles}")
        self.log(f"Updates per cycle: {n_updates_per_cycle}")
        self.log(f"Episodes per cycle: {m_episodes_per_cycle}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Buffer size: {len(self.replay_buffer)}")
        self.log(f"Device: {self.device}")
        self.log("="*70 + "\n")
        
        for cycle in range(n_cycles):
            self.log(f"\n{'='*70}")
            self.log(f"CYCLE {cycle + 1}/{n_cycles}")
            self.log(f"{'='*70}")
            
            # Phase 1: Update n lần
            self.log(f"\n[Phase 1] Training {n_updates_per_cycle} updates...")
            losses = self.train_step(n_updates_per_cycle)
            avg_loss = np.mean(losses) if losses else 0.0
            self.log(f"✓ Average loss: {avg_loss:.6f}")
            
            # Phase 2: Collect m episodes
            self.log(f"\n[Phase 2] Collecting {m_episodes_per_cycle} episodes...")
            episode_results = self.collect_episodes(m_episodes_per_cycle)
            
            rewards = [r for r, _ in episode_results]
            steps = [s for _, s in episode_results]
            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps)
            
            self.log(f"✓ Avg reward: {avg_reward:.2f}, Avg steps: {avg_steps:.1f}")
            self.log(f"  Buffer size: {len(self.replay_buffer)}, Epsilon: {self.epsilon:.4f}")
            
            # Save checkpoint
            if (cycle + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_cycle_{cycle+1}.pth')
                self.save_checkpoint(checkpoint_path)
                self.log(f"✓ Checkpoint saved: {checkpoint_path}")
        
        self.log("\n" + "="*70)
        self.log("Training Completed!")
        self.log("="*70)
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'target_agent_state_dict': self.target_agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.target_agent.load_state_dict(checkpoint['target_agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        print(f"✓ Checkpoint loaded from {path}")
