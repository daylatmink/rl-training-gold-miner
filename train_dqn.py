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
from agent.Qtention import Qtention


class ReplayBuffer:
    """Experience Replay Buffer để lưu transitions"""
    
    def __init__(self, capacity: int = 5000):
        self.buffer: Deque = deque(maxlen=capacity)
    
    def push(self, state: dict, action: int, reward: float, next_state: dict, done: bool):
        """Thêm transition vào buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Lấy random batch từ buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    """Trainer cho Deep Q-Learning"""
    
    def __init__(
        self,
        env: GoldMinerEnv,
        agent: Qtention,
        lr: float = 1e-4,
        gamma: float = 1.0,
        epsilon_start: float = 0.5,  # Thấp hơn vì chỉ có 2 actions
        epsilon_end: float = 0.01,    # End sớm hơn
        epsilon_decay: float = 0.99,  # Decay nhanh hơn, replay buffer đã giúp break correlation
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        train_freq: int = 1,  # Tần suất training: train mỗi train_freq steps
        num_planning: int = 1,  # Số lần quét buffer (planning) hoặc số batches (standard)
        use_planning: bool = True,  # True: planning approach, False: standard DQN
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.agent = agent.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.num_planning = num_planning
        self.use_planning = use_planning
        
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
    
    def select_action(self, state: dict, training: bool = True) -> tuple:
        """
        Chọn action với epsilon-greedy policy
        
        Args:
            state: Game state dict
            training: Nếu True thì dùng epsilon-greedy, False thì greedy
            
        Returns:
            (action, used_model): action được chọn và flag cho biết có dùng model không
        """
        # Check rope state để xác định forced actions
        rope_state = state['rope_state']
        is_retracting_with_item = (rope_state['state'] == 'retracting')
        is_expanding = (rope_state['state'] == 'expanding')
        is_swinging = (rope_state['state'] == 'swinging')
        has_dynamite = state['global_state']['dynamite_count'] > 0
        has_item = rope_state['has_item']
        rope_timer = rope_state.get('timer', -1)
        
        # AUTO ACTION 1: Nếu đang kéo mà không có dynamite → chỉ có action 0 (do nothing)
        if is_retracting_with_item and (not has_dynamite or not has_item):
            return 0, False  # Do nothing (không dùng model)
        
        # AUTO ACTION 2: Nếu móc đang được thả xuống (expanding) → chỉ có action 0 (do nothing)
        if is_expanding:
            return 0, False  # Do nothing (không dùng model)
        
        # AUTO ACTION 3: Nếu móc đang swinging nhưng timer > 0 (cooldown) → chỉ có action 0
        if is_swinging and rope_timer > 0:
            return 0, False  # Do nothing (không dùng model - đang cooldown)
        
        # Từ đây trở đi là các trường hợp DÙng MODEL
        if training and random.random() < self.epsilon:
            # Random exploration với Bernoulli(0.1)
            # P(action=1) = 0.1, P(action=0) = 0.9
            act = 1 if random.random() < 1.0 / 120 else 0
            if act == 1:
                self.log(f"Exploring action {act}")
            return act, True
        else:
            # Greedy exploitation
            with torch.no_grad():
                q_values = self.agent(state)  # [n_actions]
                if q_values[1] > q_values[0]:
                    self.log(f"Exploiting {q_values[0]}, {q_values[1]}")
                return q_values.argmax().item(), True
    
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
                batch = all_transitions[batch_start:batch_end]
                
                loss = self._train_on_batch(batch)
                iter_loss += loss
                num_batches += 1
            
            # Tính avg loss cho lần quét này
            avg_iter_loss = iter_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_iter_loss)
            
            # Log loss cho mỗi lần quét
            if cur_step % 600 == 0:
                self.log(f"Step {cur_step % 3600}/3600  Planning {planning_iter+1}/{self.num_planning} - Avg Loss: {avg_iter_loss:.8f} - Batches: {num_batches}/{(total_samples + self.batch_size - 1) // self.batch_size} - Buffer: {total_samples}")
        
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
            if cur_step % 600 == 0:
                self.log(f"Step {cur_step % 3600}/3600  Batch {batch_idx+1}/{self.num_planning} - Loss: {loss:.8f} - Buffer: {len(self.replay_buffer)}")
        
        return losses
    
    def     _train_on_batch(self, batch) -> float:
        """
        Train trên một batch và trả về loss
        
        Args:
            batch: List of transitions
            
        Returns:
            loss: Loss value
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute Q(s, a)
        q_values = self.agent(list(states))
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target
        with torch.no_grad():
            next_q_values = self.target_agent(list(next_states))
            next_q_values = next_q_values.max(1)[0]
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
        
        while True:
            # Select action - trả về (action, used_model)
            action, used_model = self.select_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train CHỈ KHI đạt train_freq
            
            # Update target network
            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()

            if self.total_steps % self.train_freq == 0:
                if self.total_steps % 600 == 0:
                    self.log(f"Episode step {episode_steps} reward: {episode_reward}")
                losses = self.train_step(self.total_steps)
                if losses:  # Nếu có loss (buffer đủ lớn)
                    self.losses.extend(losses)  # Thêm tất cả losses vào list
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Decay epsilon
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
                checkpoint_path = os.path.join(save_dir, f'checkpoint.pt')
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
        Evaluate agent với greedy policy
        
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
            
            while True:
                action, _ = self.select_action(state, training=False)  # Bỏ qua used_model flag
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
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
        self.log(f"Loaded checkpoint from {path}")
    
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
