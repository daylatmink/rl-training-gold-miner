"""
Warmup script để tạo replay buffer với random actions
Chơi liên tục với epsilon=1 cho đến khi buffer đạt warmup_steps
Sau đó lưu buffer vào file để train sau này load lại
"""

import torch
import pickle
import argparse
from collections import deque
from tqdm import tqdm

from model.GoldMiner import GoldMinerEnv
from agent.QCNN.Embedder import Embedder
from trainer.QcnnTrainer import angle_bins
import random


def warmup_buffer(warmup_steps: int = 1000, save_path: str = 'warmup_buffer.pkl', show: bool = False):
    """
    Chơi game với random actions để fill buffer
    
    Args:
        warmup_steps: Số transitions cần thu thập
        save_path: Đường dẫn lưu buffer
        show: Hiển thị game window (mặc định: False/headless)
    """
    print("="*60)
    print("Warmup Buffer Generation")
    print("="*60)
    print(f"Target buffer size: {warmup_steps}")
    print(f"Save path: {save_path}")
    print(f"Display mode: {'human' if show else 'headless'}")
    print("="*60)
    
    # Create environment
    print("\n[1/3] Creating environment...")
    env = GoldMinerEnv(
        render_mode='human' if show else None,
        max_steps=3600,
        levels=list(range(1, 11)),
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0,
        reward_scale=10000.0,
        game_speed=1
    )
    print("✓ Environment created")
    
    # Create buffer
    print("\n[2/3] Collecting random transitions...")
    buffer = deque(maxlen=warmup_steps * 2)  # Larger capacity để không mất data
    
    n_actions = 50
    episode = 0
    
    with tqdm(total=warmup_steps, desc="Collecting") as pbar:
        while len(buffer) < warmup_steps:
            episode += 1
            state, _ = env.reset()
            episode_steps = 0
            action_buffer = None
            state_buffer = None
            reward_buffer = 0
            angle_decision = None
            done = False
            
            while True:
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = env.step(0)
                else:
                    if angle_decision is None or done:
                        if state_buffer is not None:
                            # Save transition
                            old_env_feats, old_item_feats, old_mask = state_buffer
                            new_env_feats, new_item_feats, new_mask = Embedder.preprocess_state(state)
                            
                            buffer.append((
                                old_env_feats, old_item_feats, old_mask,
                                action_buffer, reward_buffer,
                                new_env_feats, new_item_feats, new_mask,
                                done
                            ))
                            
                            pbar.update(1)
                            episode_steps += 1
                            
                            if len(buffer) >= warmup_steps:
                                break
                        
                        if done:
                            break
                        
                        # Random action
                        action_buffer = random.randint(0, n_actions - 1)
                        angle_decision = angle_bins[action_buffer]
                        state_buffer = Embedder.preprocess_state(state)
                        reward_buffer = 0
                    
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = env.step(1)
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = env.step(0)
                
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
                
                if len(buffer) >= warmup_steps:
                    break
            
            if episode % 10 == 0:
                pbar.set_postfix({'episodes': episode, 'buffer': len(buffer)})
    
    print(f"\n✓ Collected {len(buffer)} transitions in {episode} episodes")
    
    # Save buffer
    print(f"\n[3/3] Saving buffer to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(list(buffer), f)
    print(f"✓ Buffer saved ({len(buffer)} transitions)")
    
    env.close()
    print("\n" + "="*60)
    print("Warmup completed!")
    print("="*60)
    print(f"\nTo use this buffer, load it in your trainer:")
    print(f"  with open('{save_path}', 'rb') as f:")
    print(f"      buffer_data = pickle.load(f)")
    print(f"      for transition in buffer_data:")
    print(f"          trainer.replay_buffer.buffer.append(transition)")


def load_warmup_buffer(path: str):
    """
    Load warmup buffer từ file
    
    Args:
        path: Đường dẫn file buffer
        
    Returns:
        List of transitions
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate warmup buffer for DQN training')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of transitions to collect (default: 1000)')
    parser.add_argument('--save-path', type=str, default='warmup_buffer.pkl',
                        help='Path to save buffer (default: warmup_buffer.pkl)')
    parser.add_argument('--show', action='store_true',
                        help='Show game window during warmup (default: headless)')
    
    args = parser.parse_args()
    
    warmup_buffer(
        warmup_steps=args.warmup_steps,
        save_path=args.save_path,
        show=args.show
    )
