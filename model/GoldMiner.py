"""
Gymnasium Environment cho Gold Miner Game
Theo spec trong gold_miner_env.md
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from typing import Optional, Tuple, Dict, Any
from define import set_game_speed, set_use_fixed_timestep

from entities.explosive import Explosive
import random
# Import game modules
from define import *
from scenes.game_scenes import GameScene
from state_exporter import get_game_state


class GoldMinerEnv(gym.Env):
    """
    Gymnasium Environment cho Gold Miner - Theo spec gold_miner_env.md
    
    MDP Formulation:
        - Objective: Maximize tổng điểm trong 60s
        - Episodic: Kết thúc khi time_left <= 0
        - Markov Property: State chứa đủ thông tin
        
    Action Space (Discrete(2)):
        - 0: Do nothing
        - 1: Shoot hook (nếu móc rảnh) HOẶC Use dynamite (nếu đang kéo vật phẩm)
    
    Observation Space (Dict):
        Trả về toàn bộ dict từ get_game_state() bao gồm:
        - timestamp: Thời điểm lấy state
        - global_state: score, goal, level, time_left, pause, dynamite_count
        - scene_state: level_number, current_level_id, use_generated, items_count
        - miner_state: position, speed, current_frame, state
        - rope_state: position, length, direction, state, has_item, tnt_count
        - items: List các item với type, position, size, point, direction, ranges
        - explosive_state: Trạng thái nổ (nếu có)
        
    Reward:
        - +V(item): Khi giao vật phẩm thành công
        - -C_dyna: Khi dùng dynamite (và hủy item)
        - -c_pull * g(weight): Penalty khi đang kéo vật phẩm (optional)
        - -c_step: Step cost nhỏ để tránh idle (optional)
        - 0: Các bước khác (đang bắn, chờ móc về)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 3600,  # 60 giây * 60 FPS
        levels: int | list = 1,  # Có thể là int hoặc list
        use_generated_levels: bool = True,
        c_dyna: float = 50.0,  # Cost của dynamite
        c_step: float = 0.0,  # Step cost (0 = không dùng)
        c_pull: float = 0.0,  # Penalty khi đang kéo vật phẩm (0 = không dùng)
        c_miss: float = 0.0,  # Pen1alty khi kéo miss (0 = không dùng)
        c_tnt: float = 0.0,  # Penalty khi kéo trúng TNT (0 = không dùng)
        reward_scale: float = 1000.0,  # Scale reward để ổn định training (reward_raw / reward_scale)
        game_speed: int = 1,  # Game speed: 1x, 2x, 5x, 10x (training nhanh hơn)
    ):
        super().__init__()
        self.c_tnt = c_tnt
        self.c_miss = c_miss
        self.render_mode = render_mode
        self.max_steps = max_steps
        # Nếu levels là list, lưu danh sách và sẽ sample random trong reset()
        # Nếu levels là int, chuyển thành list có 1 phần tử
        if isinstance(levels, list):
            self.levels = levels
        else:
            self.levels = [levels]
        self.current_level = self.levels[0]  # Mức hiện tại (sẽ được update trong reset)
        self.use_generated_levels = use_generated_levels
        self.c_dyna = c_dyna
        self.c_step = c_step
        self.c_pull = c_pull
        self.reward_scale = reward_scale
        self.game_speed = game_speed
        
        # Initialize Pygame
        pygame.init()
        
        # Create display
        if render_mode == "human":
            # Hiển thị cửa sổ bình thường
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Gold Miner RL Training")
        else:
            # Headless mode: tạo hidden window (không hiển thị nhưng vẫn hoạt động)
            # Sử dụng HIDDEN flag để ẩn cửa sổ thay vì dùng Surface
            os.environ['SDL_VIDEO_WINDOW_POS'] = "-3000,-3000"  # Đẩy cửa sổ ra ngoài màn hình
            self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.HIDDEN)
        
        self.clock = pygame.time.Clock()
        
        # Action space: 0=do_nothing, 1=shoot_or_dynamite
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Dict từ get_game_state()
        # Sử dụng Dict space thay vì Box để giữ nguyên cấu trúc
        self.observation_space = spaces.Dict({
            'global_state': spaces.Dict({
                'score': spaces.Box(0, np.inf, shape=(), dtype=np.float32),
                'goal': spaces.Box(0, np.inf, shape=(), dtype=np.float32),
                'level': spaces.Box(0, 100, shape=(), dtype=np.int32),
                'time_left': spaces.Box(0, 60, shape=(), dtype=np.float32),
                'pause': spaces.Discrete(2),
                'dynamite_count': spaces.Box(0, 5, shape=(), dtype=np.int32),
            }),
            'rope_state': spaces.Dict({
                'direction': spaces.Box(0, 180, shape=(), dtype=np.float32),
                'state': spaces.Discrete(3),  # swinging=0, expanding=1, retracting=2
                'length': spaces.Box(0, 1000, shape=(), dtype=np.float32),
                'has_item': spaces.Discrete(2),
            }),
            # Items sẽ được xử lý dynamic trong _get_observation
        })
        
        self.max_entities = 20
        
        # Game state
        self.game_scene = None
        self.steps = 0
        self.total_reward = 0
        self.last_score = 0
        self.done = False
        
        # Track last item being pulled để tính reward khi deliver/dynamite
        self.last_pulling_item = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment về trạng thái ban đầu
        
        Args:
            seed: Random seed
            options: Optional dict với:
                - 'level_data': Custom level data dict để load trực tiếp
        """
        super().reset(seed=seed)
        
        # Reset game state
        reset_game_state()
        reset_scaled_time()
        
        # Set game speed và enable fixed timestep cho RL training
        set_game_speed(self.game_speed)
        set_use_fixed_timestep(True)  # Enable fixed timestep cho RL training
        
        # Check for custom level from options
        custom_level_data = options.get('level_data') if options else None
        
        if custom_level_data:
            # Load custom level
            from scenes.util import level_manager
            custom_level_id = f"CUSTOM_{random.randint(10000, 99999)}"
            level_manager.level_cache[custom_level_id] = custom_level_data
            
            self.current_level = 1
            self.game_scene = GameScene(
                level=self.current_level,
                tnt=0,
                speed=1,
                is_clover=False,
                is_gem=False,
                is_rock=False,
                use_generated=False,
                time_limit=self.max_steps // 60
            )
            # Override với custom level
            from scenes.util import load_level
            self.game_scene.bg, self.game_scene.items = load_level(custom_level_id)
            self.game_scene.current_level_id = custom_level_id
        else:
            # Sample random level từ list
            self.current_level = random.choice(self.levels)
            
            # Create new game scene
            self.game_scene = GameScene(
                level=self.current_level,
                tnt=0,
                speed=1,
                is_clover=False,
                is_gem=False,
                is_rock=False,
                use_generated=self.use_generated_levels,
                time_limit=self.max_steps // 60  # Convert steps to seconds (60 FPS)
            )
        
        self.steps = 0
        self.total_reward = 0
        self.last_score = get_score()
        self.done = False
        self.last_pulling_item = None
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Thực hiện action và trả về (observation, reward, terminated, truncated, info)
        
        Args:
            action: 0=do_nothing, 1=shoot_or_dynamite
            
        Returns:
            observation: State mới
            reward: Phần thưởng theo spec
            terminated: Episode kết thúc (time <= 0)
            truncated: Episode bị cắt (max_steps)
            info: Thông tin bổ sung
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit(0)
        
        # Execute action và lấy immediate reward
        reward = self._execute_action(action)
        
        # Update game multiple times per step để physics chính xác
        # Trong game bình thường, 60 FPS = 60 updates/giây
        # Để đảm bảo physics, cần update nhiều lần giữa mỗi agent action
        frames_per_step = 1  # Số frames game chạy mỗi agent step
        
        for _ in range(frames_per_step):
            # Update game (render and update)
            self.game_scene.render(self.screen)
            self.game_scene.update(self.screen)
            
            if self.render_mode == "human":
                pygame.display.flip()
                # KHÔNG giới hạn FPS - chạy liên tục nhanh nhất có thể
                # Physics vẫn đúng vì dùng fixed timestep (1/60s)
        
        # Check score change để tính reward từ deliver
        current_score = get_score()
        score_delta = current_score - self.last_score
        
        if score_delta > 0:
            # +V(item): Giao vật phẩm thành công
            reward += score_delta
        
        self.last_score = current_score
        
        # Penalty khi móc đang bận (không phài swinging): -c_pull (cố định)
        rope = self.game_scene.rope
        if self.c_pull > 0 and rope.state != 'swinging':
            # Penalty cố định, không phụ thuộc weight
            reward -= self.c_pull
        
        # Step cost (optional)
        if self.c_step > 0:
            reward -= self.c_step
        
        # Check termination
        terminated = False
        truncated = False
        timer = self.game_scene.timer
        
        if timer <= 0:
            # Episode kết thúc khi hết thời gian
            terminated = True
            self.done = True
        
        # AUTO TERMINATE: Khi kéo hết item VÀ móc đã về (swinging)
        # Chỉ terminate khi: (1) không còn item, (2) móc đang rảnh (swinging)
        if len(self.game_scene.items) == 0 and rope.state == 'swinging':
            terminated = True
            self.done = True
        
        # Truncation: quá max_steps
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            self.done = True
        
        self.total_reward += reward
        
        # Calculate pulling penalty for info (khi móc đang bận)
        pulling_penalty = 0.0
        if self.c_pull > 0 and rope.state != 'swinging':
            pulling_penalty = self.c_pull  # Penalty cố định
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info['score_delta'] = score_delta
        info['reward_breakdown'] = {
            'score_gain': score_delta,
            'pulling_penalty': -pulling_penalty,
            'step_cost': -self.c_step if self.c_step > 0 else 0,
            'total': reward
        }
        
        # Scale reward để ổn định training
        scaled_reward = reward / self.reward_scale
        
        return obs, scaled_reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """
        Thực hiện action trong game và update state ngay lập tức
        
        Args:
            action: 0=do_nothing, 1=shoot_or_dynamite
            
        Returns:
            immediate_reward: Phần thưởng ngay lập tức
        """
        reward = 0.0
        rope = self.game_scene.rope
        miner = self.game_scene.miner
        
        if action == 0:
            # Do nothing
            pass
            
        elif action == 1:
            # Shoot hook HOẶC use dynamite
            
            if rope.state == 'swinging' and rope.timer <= 0:
                # Móc đang rảnh và không trong cooldown → bắn hook
                miner.state = 1
                # UPDATE ROPE STATE NGAY LẬP TỨC thay vì chờ rope.update()
                rope.state = 'expanding'
                
            elif rope.state == 'retracting' and rope.item is not None:
                # Đang kéo vật phẩm → sử dụng dynamite (nếu có)
                if rope.have_TNT > 0:
                    rope.is_use_TNT = True
                    miner.state = 4
                    
                    # Create explosive
                    self.game_scene.explosive = Explosive(
                        rope.x2 - 128,
                        rope.y2 - 128,
                        12
                    )
                    self.game_scene.play_Explosive = True
                    rope.have_TNT -= 1
                    set_dynamite_count(rope.have_TNT)
                    rope.length = 50
                    miner.is_TNT = True
                    
                    # -C_dyna: Cost của dynamite
                    reward -= self.c_dyna
        
        return reward
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Trả về toàn bộ game state từ get_game_state()
        
        Returns:
            state: Dict chứa toàn bộ thông tin game state
        """
        state = get_game_state(self.game_scene)
        return state
    
    def _get_info(self) -> Dict[str, Any]:
        """Lấy thông tin bổ sung về environment state"""
        if self.game_scene is None:
            return {}
        
        state = get_game_state(self.game_scene)
        
        return {
            'score': state['global_state']['score'],
            'goal': state['global_state']['goal'],
            'time_left': state['global_state']['time_left'],
            'level': state['global_state']['level'],
            'dynamite_count': state['global_state']['dynamite_count'],
            'entities_count': len(state['items']),
            'rope_state': state['rope_state']['state'],
            'total_reward': self.total_reward,
            'steps': self.steps
        }
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            # Already rendered in step()
            pass
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )
    
    def close(self):
        """Đóng environment và cleanup"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None