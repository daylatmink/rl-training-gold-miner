"""
Shopping Agent - Bandit Agent với Q-table
"""

import numpy as np
import random
from typing import Dict
from model.ShoppingEnv import get_valid_actions, decode_action, calculate_cost

class ShoppingAgent:
    """
    Shopping Agent với Q-table cho bandit decisions
    
    Q[level, action] = E[score | level, action]
    """
    
    def __init__(self, num_levels: int = 9, num_actions: int = 32):
        """
        Args:
            num_levels: Số levels (default: 9)
            num_actions: Số actions (default: 32)
        """
        self.num_levels = num_levels
        self.num_actions = num_actions
        
        # Q-table: E[r | level, action]
        # Level index: 0-8 for level 1-9
        self.Q = np.zeros((num_levels, num_actions))
        self.N = np.zeros((num_levels, num_actions))  # Visit counts
        
        # Mode and stats
        self.mode = 'train'  # 'train' or 'eval'
        self.epsilon = 0.1
        self.total_episodes = 0
    
    def set_train_mode(self, epsilon: float = 0.1):
        """Set agent to training mode với epsilon-greedy"""
        self.mode = 'train'
        self.epsilon = epsilon
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (greedy)"""
        self.mode = 'eval'
    
    def get_action(self, obs: Dict, shop_state=None, money: int = None) -> int:
        """
        Get action từ observation
        
        Args:
            obs: Observation dict với 'level'
            shop_state: ShopState (only needed for eval mode)
            money: Current money (only needed for eval mode)
            
        Returns:
            Action index (0-31)
        """
        level = obs['level']
        
        if self.mode == 'train':
            # Training: epsilon-greedy (assume infinite money)
            if random.random() < self.epsilon:
                return random.randint(0, self.num_actions - 1)
            else:
                level_idx = level - 1
                return int(np.argmax(self.Q[level_idx]))
        else:
            # Eval: greedy with constraints
            if shop_state is None or money is None:
                raise ValueError("shop_state and money required for eval mode")
            return self._select_best_action(level, shop_state, money)
    
    def _select_best_action(self, level: int, shop_state, money: int) -> int:
        """Select best action given constraints"""
        
        valid_actions = get_valid_actions(shop_state, money)
        
        if not valid_actions:
            return 0  # Skip nếu không có action nào valid
        
        level_idx = level - 1
        best_action = None
        best_d = float('-inf')
        
        for action in valid_actions:
            items = decode_action(action)
            cost = calculate_cost(items, shop_state.prices)
            expected_r = self.Q[level_idx, action]
            d_t = expected_r - cost
            
            if d_t > best_d:
                best_d = d_t
                best_action = action
        
        return best_action if best_action is not None else 0
    
    def update(self, level: int, action: int, reward: float):
        """
        Update Q-value với incremental mean
        
        Args:
            level: Level (1-9)
            action: Action index (0-31)
            reward: Observed reward (score)
        """
        level_idx = level - 1  # Convert to 0-indexed
        
        self.N[level_idx, action] += 1
        n = self.N[level_idx, action]
        
        # Incremental mean update
        self.Q[level_idx, action] += (reward - self.Q[level_idx, action]) / n
        
        self.total_episodes += 1
    
    def get_expected_reward(self, level: int, action: int) -> float:
        """Get E[r | level, action]"""
        level_idx = level - 1
        return self.Q[level_idx, action]
    
    def save(self, path: str):
        """Save Q-table"""
        np.savez(path, 
                 Q=self.Q, 
                 N=self.N, 
                 total_episodes=self.total_episodes,
                 epsilon=self.epsilon)
    
    def load(self, path: str):
        """Load Q-table"""
        data = np.load(path)
        self.Q = data['Q']
        self.N = data['N']
        self.total_episodes = int(data['total_episodes'])
        if 'epsilon' in data:
            self.epsilon = float(data['epsilon'])
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        return {
            'total_episodes': self.total_episodes,
            'q_table_shape': self.Q.shape,
            'visit_counts': self.N.sum(axis=1).tolist(),  # Per level
            'mean_q_per_level': self.Q.mean(axis=1).tolist(),
            'epsilon': self.epsilon,
        }
