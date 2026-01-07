"""
Shopping Environment cho Gold Miner Game - Bandit Formulation

Mỗi level là một Contextual Bandit với 32 arms (tổ hợp mua items).
Goal: Maximize E[d_t] = E[r_t | a_t] - cost(a_t)

Training: Assume infinite money → Learn E[r_t | a_t] cho mỗi (level, action)
Eval: Filter valid actions → Maximize d_t
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import torch

# ==================== ACTION ENCODING ====================
# 5 items -> 32 combinations (5-bit binary)
# Bit order: [rock, drink, gem, clover, dynamite]
# Arm 0  = 0b00000 = Skip all
# Arm 31 = 0b11111 = Buy all 5

ITEM_NAMES = ['rock', 'drink', 'gem', 'clover', 'dynamite']
NUM_ITEMS = 5
NUM_ACTIONS = 2 ** NUM_ITEMS  # 32

# Item indices for bit manipulation
ROCK_BIT = 4      # 0b10000 = 16
DRINK_BIT = 3     # 0b01000 = 8
GEM_BIT = 2       # 0b00100 = 4
CLOVER_BIT = 1    # 0b00010 = 2
DYNAMITE_BIT = 0  # 0b00001 = 1


@dataclass
class ShopState:
    """Trạng thái của shop tại một level"""
    level: int
    items_available: Dict[str, bool]  # {item_name: is_available}
    prices: Dict[str, int]            # {item_name: price}
    
    def get_available_mask(self) -> int:
        """Trả về bitmask của items available"""
        mask = 0
        if self.items_available.get('rock', False):
            mask |= (1 << ROCK_BIT)
        if self.items_available.get('drink', False):
            mask |= (1 << DRINK_BIT)
        if self.items_available.get('gem', False):
            mask |= (1 << GEM_BIT)
        if self.items_available.get('clover', False):
            mask |= (1 << CLOVER_BIT)
        if self.items_available.get('dynamite', False):
            mask |= (1 << DYNAMITE_BIT)
        return mask


def decode_action(action: int) -> Dict[str, bool]:
    """Decode action (0-31) thành dict of items to buy"""
    return {
        'rock': bool(action & (1 << ROCK_BIT)),
        'drink': bool(action & (1 << DRINK_BIT)),
        'gem': bool(action & (1 << GEM_BIT)),
        'clover': bool(action & (1 << CLOVER_BIT)),
        'dynamite': bool(action & (1 << DYNAMITE_BIT)),
    }


def encode_action(items: Dict[str, bool]) -> int:
    """Encode dict of items to buy thành action (0-31)"""
    action = 0
    if items.get('rock', False):
        action |= (1 << ROCK_BIT)
    if items.get('drink', False):
        action |= (1 << DRINK_BIT)
    if items.get('gem', False):
        action |= (1 << GEM_BIT)
    if items.get('clover', False):
        action |= (1 << CLOVER_BIT)
    if items.get('dynamite', False):
        action |= (1 << DYNAMITE_BIT)
    return action


def get_valid_actions(shop_state: ShopState, money: int) -> List[int]:
    """
    Trả về list các actions hợp lệ (items available AND affordable)
    
    Args:
        shop_state: Trạng thái shop hiện tại
        money: Số tiền hiện có
        
    Returns:
        List các action indices hợp lệ
    """
    valid_actions = []
    available_mask = shop_state.get_available_mask()
    
    for action in range(NUM_ACTIONS):
        # Check if action only uses available items
        if (action & ~available_mask) != 0:
            # Action yêu cầu item không available
            continue
        
        # Check if affordable
        items = decode_action(action)
        total_cost = calculate_cost(items, shop_state.prices)
        
        if total_cost <= money:
            valid_actions.append(action)
    
    return valid_actions


def calculate_cost(items: Dict[str, bool], prices: Dict[str, int]) -> int:
    """Tính tổng cost của các items được mua"""
    total = 0
    for item_name, is_buying in items.items():
        if is_buying and item_name in prices:
            total += prices[item_name]
    return total


def generate_random_shop(level: int) -> ShopState:
    """
    Generate random shop state từ game logic thực tế
    
    Sử dụng StoreScene logic để đảm bảo consistent với game
    """
    # Generate shop state giống StoreScene.__init__
    # KHÔNG thay đổi global level để tránh side effects
    items_available = {
        'rock': random.randint(0, 1) == 1,
        'drink': random.randint(0, 1) == 1,
        'gem': random.randint(0, 1) == 1,
        'clover': random.randint(0, 1) == 1,
        'dynamite': random.randint(0, 1) == 1,
    }
    
    prices = {}
    
    # Rock Collectors Book: $10-150
    if items_available['rock']:
        prices['rock'] = random.randint(10, 150)
    
    # Strength Drink: $100-400
    if items_available['drink']:
        prices['drink'] = random.randint(0, 300) + 100
    
    # Gem Polish: $200 + level*0-100
    if items_available['gem']:
        prices['gem'] = random.randint(0, level * 100) + 200
    
    # Clover: (level*2) + level*0-50 + 1
    if items_available['clover']:
        prices['clover'] = random.randint(0, level * 50) + level * 2 + 1
    
    # Dynamite: $1-301 + level*2
    if items_available['dynamite']:
        prices['dynamite'] = random.randint(0, 300) + 1 + level * 2
    
    return ShopState(level=level, items_available=items_available, prices=prices)


class ShoppingEnv(gym.Env):
    """
    Shopping Environment cho Gold Miner - Bandit Formulation
    
    Mỗi episode:
    1. Sample random level (1-9)
    2. Agent chọn action (0-31 = tổ hợp mua items)
    3. Mining agent chơi level với buffs
    4. Reward = score đạt được (không trừ cost khi training)
    
    Observation Space:
        - level: int (1-9)
        
    Action Space:
        - Discrete(32): 32 tổ hợp mua items
        
    Reward:
        - Training: E[r_t | a_t] = score từ mining agent
        - Eval: d_t = score - cost (handled bên ngoài)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        mining_env,  # GoldMinerEnv instance
        mining_trainer,  # Mining trainer (có method select_action)
        levels: List[int] = None,
    ):
        """
        Args:
            mining_env: GoldMinerEnv instance đã khởi tạo
            mining_trainer: Mining trainer instance (QtentionTrainer, QcnnTrainer, etc.)
            levels: List levels để sample (default: 1-9)
        """
        super().__init__()
        
        self.mining_env = mining_env
        self.mining_trainer = mining_trainer
        self.levels = levels if levels else list(range(1, 10))  # 1-9
        
        # Action space: 32 combinations
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # Observation space: level (1-9)
        self.observation_space = spaces.Dict({
            'level': spaces.Discrete(10),  # 0-9, chỉ dùng 1-9
        })
        
        # State
        self.current_level = 1
        self.shop_state = None
        self.done = False
        
        # Statistics
        self.episode_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment - sample new level
        
        Args:
            seed: Random seed
            options: Optional dict với:
                - 'level': Specific level to use
                
        Returns:
            observation: {'level': int}
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Store seed for mining episode
        self.current_seed = seed
        
        # Sample level
        if options and 'level' in options:
            self.current_level = options['level']
        else:
            self.current_level = random.choice(self.levels)
        
        # Generate random shop (for reference, không dùng khi training với infinite money)
        self.shop_state = generate_random_shop(self.current_level)
        
        self.done = False
        self.episode_count += 1
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Thực hiện action và chạy mining để lấy reward
        
        Args:
            action: 0-31 (tổ hợp items mua)
            
        Returns:
            observation: State mới (level)
            reward: Score từ mining agent
            done: True (bandit = 1 step episode)
            info: Chi tiết về episode
        """
        if self.done:
            return self._get_observation(), 0.0, True, self._get_info()
        
        # Decode action to buffs
        items = decode_action(action)
        
        # Run mining with buffs và get score
        score = self._run_mining_episode(items)
        
        # Reward = raw score (không trừ cost khi training)
        reward = score
        
        # Bandit: Episode kết thúc sau 1 action
        self.done = True
        
        obs = self._get_observation()
        info = self._get_info()
        info['action'] = action
        info['items_bought'] = items
        info['raw_score'] = score
        info['shop_state'] = {
            'items_available': self.shop_state.items_available,
            'prices': self.shop_state.prices,
        }
        
        # Tính cost nếu cần (cho logging)
        cost = calculate_cost(items, self.shop_state.prices)
        info['cost'] = cost
        info['d_t'] = score - cost  # Net reward
        
        return obs, reward, True, info
    
    def _run_mining_episode(self, buffs: Dict[str, bool]) -> float:
        """
        Chạy 1 episode mining với buffs và trả về total score
        
        Args:
            buffs: Dict of buffs {rock, drink, gem, clover, dynamite}
            
        Returns:
            Total score đạt được
        """
        from define import reset_game_state, get_score, set_level
        import random
        import numpy as np
        import torch
        
        # Use stored seed for mining episode
        if hasattr(self, 'current_seed') and self.current_seed is not None:
            random.seed(self.current_seed)
            np.random.seed(self.current_seed)
            torch.manual_seed(self.current_seed)
        
        # Reset game state
        reset_game_state()
        
        # Set global level to ensure correct level
        set_level(self.current_level)
        
        # Reset mining env để chạy level mới with correct level and seed
        obs, info = self.mining_env.reset(seed=self.current_seed, options={'level': self.current_level})
        
        # Apply buffs trực tiếp lên game_scene sau khi reset
        game_scene = self.mining_env.game_scene
        
        # Apply dynamite buff
        if buffs.get('dynamite', False):
            from define import get_dynamite_count, set_dynamite_count
            current_tnt = get_dynamite_count()
            set_dynamite_count(min(current_tnt + 1, 5))
            game_scene.rope.have_TNT = min(game_scene.rope.have_TNT + 1, 5)
        
        # Apply drink buff (speed)
        if buffs.get('drink', False):
            game_scene.rope.buff_speed = 2
        
        # Apply gem, clover, rock buffs bằng cách reload items
        is_gem = buffs.get('gem', False)
        is_clover = buffs.get('clover', False)
        is_rock = buffs.get('rock', False)
        
        if is_gem or is_clover or is_rock:
            from scenes.util import load_level
            # Reload level với buffs
            level_id = game_scene.current_level_id
            game_scene.bg, game_scene.items = load_level(level_id, is_clover, is_gem, is_rock)
        
        # Run episode KHÔNG dùng trainer.evaluate() vì nó sẽ reset lại env với level random
        # Thay vào đó, chạy trực tiếp episode loop với env đã reset sẵn với level đúng
        from trainer.QtentionTrainer import angle_bins
        
        self.mining_trainer.agent.eval()
        
        # Get initial state từ env đã được reset ở trên
        state = self.mining_env._get_observation()
        
        episode_reward = 0.0
        action_buffer = None
        reward_buffer = 0
        angle_decision = None
        done = False
        prev_total_points = 0.0
        miss_streak = 0
        prev_num_items = -1
        
        with torch.no_grad():
            while True:
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = self.mining_env.step(0)
                else:
                    if angle_decision is None or done:
                        if action_buffer is not None:
                            cur_total_points = sum(item['point'] for item in state['items'])
                            if reward_buffer == 0 and cur_total_points < prev_total_points:
                                lost_points = prev_total_points - cur_total_points
                                tnt_penalty = -self.mining_env.c_tnt * lost_points / self.mining_env.reward_scale
                                reward_buffer += tnt_penalty
                            prev_total_points = cur_total_points
                            
                            cur_num_items = len(state['items'])
                            if cur_num_items == prev_num_items:
                                miss_streak += 1
                            else:
                                miss_streak = 0
                            prev_num_items = cur_num_items
                            
                            episode_reward += reward_buffer
                            reward_buffer = 0
                            action_buffer = None
                        if done:
                            break
                        action_buffer, _ = self.mining_trainer.select_action(state, miss_streak=miss_streak, training=False)
                        angle_decision = angle_bins[action_buffer]
                    
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = self.mining_env.step(1)
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = self.mining_env.step(0)
        
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
        
        self.mining_trainer.agent.train()
        
        # Get final score
        total_score = get_score()
        
        return total_score
    
    def _get_observation(self) -> Dict:
        """Trả về observation"""
        return {
            'level': self.current_level,
        }
    
    def _get_info(self) -> Dict:
        """Trả về info"""
        return {
            'level': self.current_level,
            'episode_count': self.episode_count,
        }
    
    def get_valid_actions(self, money: int) -> List[int]:
        """
        Trả về list actions hợp lệ với shop state hiện tại và số tiền
        
        Args:
            money: Số tiền hiện có
            
        Returns:
            List action indices hợp lệ
        """
        if self.shop_state is None:
            return [0]  # Skip only
        return get_valid_actions(self.shop_state, money)
    
    def calculate_d_t(self, action: int, expected_score: float) -> float:
        """
        Tính d_t = E[r_t | a_t] - cost(a_t)
        
        Args:
            action: Action index
            expected_score: E[r_t | a_t] from Q-table
            
        Returns:
            d_t value
        """
        if self.shop_state is None:
            return expected_score
        
        items = decode_action(action)
        cost = calculate_cost(items, self.shop_state.prices)
        return expected_score - cost
    
    def render(self):
        """Render - delegate to mining env"""
        return self.mining_env.render()
    
    def close(self):
        """Cleanup - không close mining_env vì được quản lý bên ngoài"""
        pass


# ==================== UTILITY FUNCTIONS ====================

def action_to_string(action: int) -> str:
    """Convert action to readable string"""
    items = decode_action(action)
    bought = [name for name, is_buying in items.items() if is_buying]
    if not bought:
        return "Skip"
    return "+".join(bought)
