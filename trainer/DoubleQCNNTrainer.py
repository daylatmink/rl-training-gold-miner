"""
Double Deep Q-Learning Training cho Gold Miner với QCNN Network
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
from agent.QCNN.QCNN import QCNN
from agent.QCNN.Embedder import Embedder
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
    """Experience Replay Buffer (QCNN-format)."""

    def __init__(self, capacity: int = 5000, device='cpu', n_actions: int = 50):
        self.buffer: Deque = deque(maxlen=capacity)
        self.device = device
        self.n_actions = n_actions
        # Track action counts cho inverse frequency exploration
        self.action_counts = np.zeros(n_actions, dtype=np.float32)

    def push(
        self,
        env_feats: torch.Tensor, items_feats: torch.Tensor, mask: torch.Tensor,
        action: int, reward: float,
        next_env_feats: torch.Tensor, next_items_feats: torch.Tensor, next_mask: torch.Tensor,
        done: bool
    ):
        """
        Lưu transition (CPU tensors).
        env_feats: [10]
        items_feats: [max_items, 23]
        mask: [max_items] (1 real, 0 pad)
        """
        
        if len(self.buffer) == self.buffer.maxlen:
            old_action = self.buffer[0][3]  # action ở index 3
            self.action_counts[old_action] = max(0, self.action_counts[old_action] - 1)
        
        self.buffer.append((
            env_feats, items_feats, mask,
            action, reward,
            next_env_feats, next_items_feats, next_mask,
            done
        ))
        
        self.action_counts[action] += 1

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        env_feats, items_feats, masks, actions, rewards, next_env_feats, next_items_feats, next_masks, dones = zip(*batch)

        return {
            "env_feats": torch.stack(env_feats, dim=0),                 # [B,10]
            "items_feats": torch.stack(items_feats, dim=0),             # [B,M,23]
            "masks": torch.stack(masks, dim=0),                         # [B,M]
            "actions": torch.tensor(actions, dtype=torch.long),         # [B]
            "rewards": torch.tensor(rewards, dtype=torch.float32),      # [B]
            "next_env_feats": torch.stack(next_env_feats, dim=0),       # [B,10]
            "next_items_feats": torch.stack(next_items_feats, dim=0),   # [B,M,23]
            "next_masks": torch.stack(next_masks, dim=0),               # [B,M]
            "dones": torch.tensor(dones, dtype=torch.float32),          # [B]
        }

    def __len__(self) -> int:
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


class DoubleQCNNTrainer:
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
        agent: QCNN,
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
        self.target_agent = QCNN().to(device)
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
        # Random generator riêng cho selective greedy (để có thể reset seed độc lập)
        self.selective_rng = random.Random()
    
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
        self.logger = logging.getLogger('DoubleQCNNTrainer')
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
    
    @torch.no_grad()
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
        env_feats, items_feats, mask = Embedder.preprocess_state(state)
        
        # Convert to torch tensors và chuyển sang device
        type_ids_t = env_feats.unsqueeze(0).to(self.device)            # [1, 10]
        item_feats_t = items_feats.unsqueeze(0).to(self.device)        # [1, M, 23]
        mask_t = mask.unsqueeze(0).to(self.device)                  # [1, M]
        
        # Sau khi kéo hụt: random k actions và chọn max trong k đó (ưu tiên hơn epsilon-greedy)
        if miss_streak > 0:
            with torch.no_grad():
                q_values = self.agent(type_ids_t, item_feats_t, mask_t)  # [1, n_actions]
                # Random k action indices (dùng selective_rng riêng để reproducible)
                random_actions = self.selective_rng.sample(range(self.agent.n_actions), min(k_random, self.agent.n_actions))
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
                q_values = self.agent(type_ids_t, item_feats_t, mask_t)  # [1, n_actions]
                action = q_values.argmax(dim=1).item()
                q_value = q_values[0][action].cpu().item()
            used_model = True
            action_mode = 'model'
        
        # Lưu thông tin action vào global state để hiển thị trên màn hình
        from define import set_ai_action_info
        set_ai_action_info(action, q_value, used_model, action_mode)
        
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
            
            # Forward pass - Online network
            q_values = self.agent(
                batch['env_feats'].cuda(), batch['items_feats'].cuda(),
                batch['masks'].cuda()
            )  # [B, n_actions]
            
            # Q(s, a)
            q_sa = q_values.gather(1, batch['actions'].unsqueeze(1).cuda()).squeeze(1)  # [B]
            
            # Double DQN target
            with torch.no_grad():
                # Online network chọn action
                next_q_online = self.agent(
                    batch['next_env_feats'].cuda(), batch['next_items_feats'].cuda(),
                    batch['next_masks'].cuda()
                )  # [B, n_actions]
                best_next_actions = next_q_online.argmax(dim=1)  # [B]
                
                # Target network đánh giá action
                next_q_target = self.target_agent(
                    batch['next_env_feats'].cuda(), batch['next_items_feats'].cuda(),
                    batch['next_masks'].cuda()
                )  # [B, n_actions]
                next_q = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)  # [B]
                
                # Target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
                target = batch['rewards'].cuda() + self.gamma * next_q * (1 - batch['dones'].cuda())
            
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
                            old_env_feats, old_items_feats, old_mask = state_buffer
                            
                            # Preprocess new state
                            new_env_feats, new_items_feats, new_mask = Embedder.preprocess_state(
                                state, max_items=self.max_items
                            )
                            
                            # Push to buffer
                            self.replay_buffer.push(
                                old_env_feats, old_items_feats, old_mask,
                                action_buffer, reward_buffer,
                                new_env_feats, new_items_feats, new_mask,
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
                        env_feats, items_feats, mask = Embedder.preprocess_state(state, max_items=self.max_items)
                        
                        state_buffer = (env_feats, items_feats, mask)
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
        self.log("Double QCNN Training Started")
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