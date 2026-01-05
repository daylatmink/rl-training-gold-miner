"""
Trainer cho Shopping Bandit - Training loop logic
"""

import torch
import numpy as np
from typing import Dict
from tqdm import tqdm


class ShoppingBanditTrainer:
    """
    Trainer cho Shopping Bandit - Chỉ lo training loop
    
    Agent (ShoppingAgent) chứa Q-table
    Trainer chỉ lo: training loop, logging, epsilon scheduling
    """
    
    def __init__(self, env, agent):
        """
        Args:
            env: ShoppingEnv instance
            agent: ShoppingAgent instance
        """
        self.env = env
        self.agent = agent
    
    def train(
        self, 
        num_episodes: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        save_freq: int = 100,
        save_path: str = None,
    ):
        """
        Training loop
        
        Args:
            num_episodes: Số episodes để train
            epsilon_start: Starting epsilon
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            save_freq: Frequency để save checkpoint
            save_path: Path để save (nếu None, không save)
        """
        epsilon = epsilon_start
        
        episode_rewards = []
        
        try:
            # Disable gradient tracking for mining agent (faster)
            with torch.no_grad():
                for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
                    # Set epsilon
                    self.agent.set_train_mode(epsilon=epsilon)
                    
                    # Reset env
                    obs, info = self.env.reset()
                    
                    # Get action
                    action = self.agent.get_action(obs)
                    
                    # Step
                    obs, reward, done, info = self.env.step(action)
                    
                    # Update Q-table
                    level = info['level']
                    self.agent.update(level, action, reward)
                    
                    episode_rewards.append(reward)
                    
                    # Decay epsilon
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)
                    
                    # Logging
                    if episode % 100 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        print(f"\nEpisode {episode}/{num_episodes}")
                        print(f"  Avg Reward (last 100): {avg_reward:.1f}")
                        print(f"  Epsilon: {epsilon:.4f}")
                        print(f"  Total episodes: {self.agent.total_episodes}")
                    
                    # Save checkpoint
                    if save_path and episode % save_freq == 0:
                        checkpoint_path = f"{save_path}_ep_{episode}.npz"
                        self.agent.save(checkpoint_path)
                        print(f"  Saved: {checkpoint_path}")
        
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("TRAINING INTERRUPTED BY USER")
            print("="*60)
            
            # Save checkpoint when interrupted
            if save_path:
                interrupt_path = f"{save_path}_interrupted_ep_{len(episode_rewards)}.npz"
                self.agent.save(interrupt_path)
                print(f"✓ Checkpoint saved: {interrupt_path}")
                print(f"  Episodes completed: {len(episode_rewards)}")
                print(f"  Current epsilon: {epsilon:.4f}")
            
            print("="*60 + "\n")
        
        return episode_rewards
    
    def evaluate(
        self,
        num_episodes: int,
        shop_states_and_money: list  # List of (shop_state, money) tuples
    ) -> Dict:
        """
        Evaluate agent với constraints
        
        Args:
            num_episodes: Số episodes
            shop_states_and_money: List of (shop_state, money) for each episode
            
        Returns:
            Statistics dict
        """
        self.agent.set_eval_mode()
        
        episode_rewards = []
        episode_d_ts = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            
            # Get shop state and money for this episode
            shop_state, money = shop_states_and_money[episode % len(shop_states_and_money)]
            
            # Get action with constraints
            action = self.agent.get_action(obs, shop_state=shop_state, money=money)
            
            # Step
            obs, reward, done, info = self.env.step(action)
            
            episode_rewards.append(reward)
            episode_d_ts.append(info['d_t'])
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'avg_d_t': np.mean(episode_d_ts),
            'rewards': episode_rewards,
        }
