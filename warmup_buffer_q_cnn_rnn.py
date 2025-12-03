"""
Warmup script để tạo replay buffer cho QCnnRnn với episode-based storage
Mỗi sample là một episode hoàn chỉnh với sequence length được pad về max_steps=15
"""

import torch
import pickle
import argparse
from collections import deque
from tqdm import tqdm

from model.GoldMiner import GoldMinerEnv
from agent.QCnnRnn.Embedder import Embedder
from trainer.QcnnTrainer import angle_bins
import random


def warmup_buffer(warmup_episodes: int = 100, max_steps: int = 15, max_items: int = 30,
                  save_path: str = 'warmup_buffer_rnn.pkl', show: bool = False):
    """
    Chơi game với random actions để fill buffer với episode-based storage
    
    Mỗi episode được lưu dưới dạng:
        - env_feats: [T, 10] (padded to max_steps)
        - item_feats: [T, max_items, 23] (padded to max_steps and max_items)
        - masks: [T, max_items] (padded to max_steps)
        - actions: [T-1] (padded to max_steps-1)
        - rewards: [T-1] (padded to max_steps-1)
        - seq_len: int (actual sequence length before padding)
        
    Args:
        warmup_episodes: Số episodes cần thu thập
        max_steps: Max sequence length (T) để pad
        max_items: Max items (L) để pad
        save_path: Đường dẫn lưu buffer
        show: Hiển thị game window (mặc định: False/headless)
    """
    print("="*60)
    print("Warmup Buffer Generation (Episode-based for QCnnRnn)")
    print("="*60)
    print(f"Target episodes: {warmup_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Max items: {max_items}")
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
    
    # Create buffer (list of episodes)
    print(f"\n[2/3] Collecting {warmup_episodes} random episodes...")
    buffer = []
    
    n_actions = 50
    
    with tqdm(total=warmup_episodes, desc="Episodes") as pbar:
        for episode_idx in range(warmup_episodes):
            state, _ = env.reset()
            
            # Storage for this episode
            env_feats_list = []
            item_feats_list = []
            masks_list = []
            actions_list = []
            rewards_list = []
            
            action_buffer = len(angle_bins)
            angle_decision = None
            done = False
            reward_buffer = 0
            
            # Collect one episode
            while True:
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] 
                                or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = env.step(0)
                else:
                    if angle_decision is None or done:
                        # Save current state s_t
                        env_f, item_f, mask = Embedder.preprocess_state(state, max_items=max_items)
                        env_feats_list.append(env_f)
                        item_feats_list.append(item_f)
                        masks_list.append(mask)
                        
                        # Save accumulated reward r_t from previous action (if exists)
                        rewards_list.append(reward_buffer)
                        actions_list.append(action_buffer)
                        
                        if done:
                            break
                        
                        # Select random action a_t
                        action_buffer = random.randint(0, n_actions - 1)
                        angle_decision = angle_bins[action_buffer]
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
            
            # Pad episode to max_steps
            seq_len = len(env_feats_list)
            
            if seq_len > 0:
                # Stack states
                env_feats_tensor = torch.stack(env_feats_list)  # [seq_len, 10]
                item_feats_tensor = torch.stack(item_feats_list)  # [seq_len, max_items, 23]
                masks_tensor = torch.stack(masks_list)  # [seq_len, max_items]
                
                # Convert actions and rewards to tensors
                actions_tensor = torch.tensor(actions_list, dtype=torch.long)  # [seq_len-1]
                rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)  # [seq_len-1]
                
                # Pad to max_steps
                if seq_len < max_steps:
                    pad_len = max_steps - seq_len
                    
                    # Pad env_feats
                    env_pad = torch.zeros(pad_len, 10, dtype=torch.float32)
                    env_feats_tensor = torch.cat([env_feats_tensor, env_pad], dim=0)
                    
                    # Pad item_feats
                    item_pad = torch.zeros(pad_len, max_items, 23, dtype=torch.float32)
                    item_feats_tensor = torch.cat([item_feats_tensor, item_pad], dim=0)
                    
                    # Pad masks
                    mask_pad = torch.zeros(pad_len, max_items, dtype=torch.float32)
                    masks_tensor = torch.cat([masks_tensor, mask_pad], dim=0)
                    
                    # Pad actions max_steps
                    action_pad_len = max_steps - len(actions_list)
                    if action_pad_len > 0:
                        action_pad = torch.zeros(action_pad_len, dtype=torch.long)
                        actions_tensor = torch.cat([actions_tensor, action_pad], dim=0)
                    
                    # Pad rewards
                    reward_pad_len = max_steps - len(rewards_list)
                    if reward_pad_len > 0:
                        reward_pad = torch.zeros(reward_pad_len, dtype=torch.float32)
                        rewards_tensor = torch.cat([rewards_tensor, reward_pad], dim=0)
                
                # Save episode
                episode_data = {
                    'env_feats': env_feats_tensor,  # [max_steps, 10]
                    'item_feats': item_feats_tensor,  # [max_steps, max_items, 23]
                    'masks': masks_tensor,  # [max_steps, max_items]
                    'actions': actions_tensor,  # [max_steps-1]
                    'rewards': rewards_tensor,  # [max_steps-1]
                    'seq_len': seq_len  # Actual length before padding
                }
                
                buffer.append(episode_data)
            
            pbar.update(1)
            pbar.set_postfix({'avg_steps': sum(ep['seq_len'] for ep in buffer) / len(buffer) if buffer else 0})
    
    print(f"\n✓ Collected {len(buffer)} episodes")
    avg_steps = sum(ep['seq_len'] for ep in buffer) / len(buffer)
    print(f"  Average episode length: {avg_steps:.2f} steps")
    
    # Save buffer
    print(f"\n[3/3] Saving buffer to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"✓ Buffer saved ({len(buffer)} episodes)")
    
    env.close()
    print("\n" + "="*60)
    print("Warmup completed!")
    print("="*60)
    print(f"\nTo use this buffer, load it in your trainer:")
    print(f"  with open('{save_path}', 'rb') as f:")
    print(f"      buffer_data = pickle.load(f)")
    print(f"      for episode in buffer_data:")
    print(f"          trainer.replay_buffer.push(episode)")


def load_warmup_buffer(path: str):
    """
    Load warmup buffer từ file
    
    Args:
        path: Đường dẫn file buffer
        
    Returns:
        List of episode dicts
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episode-based warmup buffer for QCnnRnn training')
    parser.add_argument('--warmup-episodes', type=int, default=100,
                        help='Number of episodes to collect (default: 100)')
    parser.add_argument('--max-steps', type=int, default=15,
                        help='Max sequence length per episode (default: 15)')
    parser.add_argument('--max-items', type=int, default=30,
                        help='Max items to pad (default: 30)')
    parser.add_argument('--save-path', type=str, default='warmup_buffer_rnn.pkl',
                        help='Path to save buffer (default: warmup_buffer_rnn.pkl)')
    parser.add_argument('--show', action='store_true',
                        help='Show game window during warmup (default: headless)')
    
    args = parser.parse_args()
    
    warmup_buffer(
        warmup_episodes=args.warmup_episodes,
        max_steps=args.max_steps,
        max_items=args.max_items,
        save_path=args.save_path,
        show=args.show
    )
