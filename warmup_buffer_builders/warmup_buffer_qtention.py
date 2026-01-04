"""
Warmup script để tạo replay buffer với random actions cho Qtention network.
Sử dụng Embedder.preprocess_state() để preprocess state thành tensors.
Chơi liên tục với epsilon=1 cho đến khi buffer đạt warmup_steps.
Sau đó lưu buffer vào file để train sau này load lại.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import argparse
from collections import deque
from tqdm import tqdm

from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Embedder import Embedder
from trainer.QcnnTrainer import angle_bins
import random


def pad_state_tensors(type_ids, item_feats, mov_idx, mov_feats, max_length):
    """
    Pad state tensors về max_length.
    
    Args:
        type_ids: Tensor [L] - Token type IDs
        item_feats: Tensor [L, 10] - Item features
        mov_idx: Tensor [l] - Movement indices
        mov_feats: Tensor [l, 3] - Movement features
        max_length: Target length (ENV + max_items)
    
    Returns:
        Padded tensors với cùng shapes + actual_length (số lượng items thực tế trước khi pad)
    """
    current_length = len(type_ids)
    actual_length = current_length
    
    if current_length < max_length:
        pad_size = max_length - current_length
        
        # Pad type_ids với PAD token (10)
        pad_type_ids = torch.full((pad_size,), Embedder.TOKEN_TYPES['PAD'], dtype=torch.int64)
        type_ids = torch.cat([type_ids, pad_type_ids])
        
        # Pad item_feats với zeros
        pad_feats = torch.zeros((pad_size, 10), dtype=torch.float32)
        item_feats = torch.cat([item_feats, pad_feats])
    elif current_length > max_length:
        # Truncate nếu vượt quá (không nên xảy ra nếu max_items được set đúng)
        type_ids = type_ids[:max_length]
        item_feats = item_feats[:max_length]
        actual_length = max_length
    
    return type_ids, item_feats, mov_idx, mov_feats, actual_length


def warmup_buffer(warmup_steps: int = 1000, save_path: str = 'warmup_buffer_qtention.pkl', show: bool = False, max_items: int = 30):
    """
    Chơi game với random actions để fill buffer cho Qtention network.
    
    Args:
        warmup_steps: Số transitions cần thu thập
        save_path: Đường dẫn lưu buffer
        show: Hiển thị game window (mặc định: False/headless)
        max_items: Max số items để pad về (mặc định: 30)
    """
    print("="*60)
    print("Warmup Buffer Generation for Qtention Network")
    print("="*60)
    print(f"Target buffer size: {warmup_steps}")
    print(f"Max items (padded): {max_items}")
    print(f"Max sequence length: {max_items + 1} (1 ENV + {max_items} items)")
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
                            # Save transition - preprocess states using Embedder
                            old_type_ids, old_item_feats, old_mov_idx, old_mov_feats = state_buffer
                            new_type_ids, new_item_feats, new_mov_idx, new_mov_feats = Embedder.preprocess_state(state, max_items=max_items)
                            
                            # Pad to max_length
                            max_length = max_items + 1  # ENV + items
                            old_type_ids, old_item_feats, old_mov_idx, old_mov_feats, old_length = pad_state_tensors(
                                old_type_ids, old_item_feats, old_mov_idx, old_mov_feats, max_length
                            )
                            new_type_ids, new_item_feats, new_mov_idx, new_mov_feats, new_length = pad_state_tensors(
                                new_type_ids, new_item_feats, new_mov_idx, new_mov_feats, max_length
                            )
                            
                            buffer.append((
                                old_type_ids, old_item_feats, old_mov_idx, old_mov_feats, old_length,
                                action_buffer, reward_buffer,
                                new_type_ids, new_item_feats, new_mov_idx, new_mov_feats, new_length,
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
                        state_buffer = Embedder.preprocess_state(state, max_items=max_items)
                        reward_buffer = 0
                    
                    # Execute action
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


def verify_buffer(path: str):
    """
    Verify buffer format và in thông tin cơ bản
    
    Args:
        path: Đường dẫn file buffer
    """
    print(f"Loading buffer from {path}...")
    buffer = load_warmup_buffer(path)
    
    print(f"\n{'='*60}")
    print("Buffer Information")
    print(f"{'='*60}")
    print(f"Total transitions: {len(buffer)}")
    
    if len(buffer) > 0:
        transition = buffer[0]
        print(f"\nTransition format:")
        print(f"  - Old state: (type_ids, item_feats, mov_idx, mov_feats, actual_length)")
        print(f"    • type_ids shape: {transition[0].shape}")
        print(f"    • item_feats shape: {transition[1].shape}")
        print(f"    • mov_idx shape: {transition[2].shape}")
        print(f"    • mov_feats shape: {transition[3].shape}")
        print(f"    • actual_length: {transition[4]} (int)")
        print(f"  - Action: {transition[5]} (int)")
        print(f"  - Reward: {transition[6]:.2f} (float)")
        print(f"  - New state: (type_ids, item_feats, mov_idx, mov_feats, actual_length)")
        print(f"    • type_ids shape: {transition[7].shape}")
        print(f"    • item_feats shape: {transition[8].shape}")
        print(f"    • mov_idx shape: {transition[9].shape}")
        print(f"    • mov_feats shape: {transition[10].shape}")
        print(f"    • actual_length: {transition[11]} (int)")
        print(f"  - Done: {transition[12]} (bool)")
        
        # Statistics
        actions = [t[5] for t in buffer]
        rewards = [t[6] for t in buffer]
        dones = [t[12] for t in buffer]
        old_lengths = [t[4] for t in buffer]
        new_lengths = [t[11] for t in buffer]
        
        print(f"\nStatistics:")
        print(f"  - Unique actions: {len(set(actions))}")
        print(f"  - Action range: [{min(actions)}, {max(actions)}]")
        print(f"  - Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        print(f"  - Average reward: {sum(rewards)/len(rewards):.2f}")
        print(f"  - Done transitions: {sum(dones)} ({sum(dones)/len(dones)*100:.1f}%)")
        
        # Check sequence lengths (padded)
        seq_lengths = [len(t[0]) for t in buffer]
        print(f"  - Padded sequence length: {seq_lengths[0]} (all same)")
        
        # Check actual lengths (before padding)
        print(f"  - Actual length range: [{min(old_lengths)}, {max(old_lengths)}]")
        print(f"  - Average actual length: {sum(old_lengths)/len(old_lengths):.1f}")
        
        # Check movement items
        mov_counts = [len(t[2]) for t in buffer]
        print(f"  - Movement items range: [{min(mov_counts)}, {max(mov_counts)}]")
        print(f"  - Average movement items: {sum(mov_counts)/len(mov_counts):.1f}")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate warmup buffer for Qtention network training')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of transitions to collect (default: 1000)')
    parser.add_argument('--save-path', type=str, default='buffers/warmup_buffer_qtention.pkl',
                        help='Path to save buffer (default: buffers/warmup_buffer_qtention.pkl)')
    parser.add_argument('--max-items', type=int, default=30,
                        help='Maximum number of items to pad to (default: 30)')
    parser.add_argument('--show', action='store_true',
                        help='Show game window during warmup (default: headless)')
    parser.add_argument('--verify', type=str, default=None,
                        help='Verify and show information about an existing buffer file')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_buffer(args.verify)
    else:
        warmup_buffer(
            warmup_steps=args.warmup_steps,
            save_path=args.save_path,
            show=args.show,
            max_items=args.max_items
        )
