"""
Warmup script để tạo replay buffer cho QCnnRnn với episode-based storage
Sử dụng Trainer để thu thập episodes với random policy
"""

import torch
import pickle
import argparse
from tqdm import tqdm

from model.GoldMiner import GoldMinerEnv
from agent.QCnnRnn.QCnnRnn import QCnnRnn
from trainer.DoubleQCnnRnnTrainer import DoubleQCnnRnnTrainer


def warmup_buffer(warmup_episodes: int = 100, save_path: str = 'warmup_buffer_rnn.pkl', show: bool = False):
    """
    Sử dụng Trainer để thu thập episodes với random policy (epsilon=1.0)
    
    Args:
        warmup_episodes: Số episodes cần thu thập
        save_path: Đường dẫn lưu buffer
        show: Hiển thị game window (mặc định: False/headless)
    """
    print("="*60)
    print("Warmup Buffer Generation (Using DoubleQCnnRnnTrainer)")
    print("="*60)
    print(f"Target episodes: {warmup_episodes}")
    print(f"Save path: {save_path}")
    print(f"Display mode: {'human' if show else 'headless'}")
    print("="*60)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = GoldMinerEnv(
        render_mode='human' if show else None,
        max_steps=3600,
        levels=list(range(1, 11)),
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0,
        c_miss=10.0,
        c_tnt=0.3,
        reward_scale=10000.0,
        game_speed=1
    )
    print("✓ Environment created")
    
    # Create agent
    print("\n[2/4] Creating agent...")
    agent = QCnnRnn(
        d_model=24,
        n_actions=50,
        d_hidden=48,
        dropout=0.2,
        max_streak=15,
        num_layers=1
    )
    print("✓ Agent created")
    
    # Create trainer with epsilon=1.0 (pure random policy)
    print("\n[3/4] Creating trainer with random policy...")
    trainer = DoubleQCnnRnnTrainer(
        env=env,
        agent=agent,
        lr=3e-4,
        gamma=0.9,
        epsilon_start=1.0,  # 100% random actions
        epsilon_end=1.0,
        epsilon_decay=1.0,
        buffer_size=warmup_episodes,
        batch_size=64,
        target_update_freq=1000,
        train_freq=1,
        num_planning=1,
        use_planning=False,
        warmup_steps=0,  # Không cần warmup vì đã là random
        top_k=-1
    )
    print("✓ Trainer created")
    
    # Collect episodes using trainer
    print(f"\n[4/4] Collecting {warmup_episodes} random episodes...")
    with tqdm(total=warmup_episodes, desc="Episodes") as pbar:
        for episode_idx in range(warmup_episodes):
            total_reward, steps = trainer.train_episode()
            pbar.update(1)
            pbar.set_postfix({'steps': steps, 'reward': f'{total_reward:.2f}'})
    
    # Get buffer from trainer
    buffer = trainer.replay_buffer.buffer
    
    print(f"\n✓ Collected {len(buffer)} episodes")
    if buffer:
        avg_steps = sum(ep[5] for ep in buffer) / len(buffer)  # ep[5] is seq_len
        print(f"  Average episode length: {avg_steps:.2f} steps")
    
    # Save buffer
    print(f"\nSaving buffer to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"✓ Buffer saved ({len(buffer)} episodes)")
    
    env.close()
    print("\n" + "="*60)
    print("Warmup completed!")
    print("="*60)
    print(f"\nTo use this buffer in main_q_trainer.py:")
    print(f"  python main_q_trainer.py --net cnn_rnn --strategy DoubleQ --warmup-path {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episode-based warmup buffer for QCnnRnn using Trainer')
    parser.add_argument('--warmup-episodes', type=int, default=100,
                        help='Number of episodes to collect (default: 100)')
    parser.add_argument('--save-path', type=str, default='warmup_buffer_rnn.pkl',
                        help='Path to save buffer (default: warmup_buffer_rnn.pkl)')
    parser.add_argument('--show', action='store_true',
                        help='Show game window during warmup (default: headless)')
    
    args = parser.parse_args()
    
    warmup_buffer(
        warmup_episodes=args.warmup_episodes,
        save_path=args.save_path,
        show=args.show
    )
