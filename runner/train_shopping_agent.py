"""
Training script cho Shopping Agent

Train bandit agent để học mua items tối ưu ở shop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from datetime import datetime

from model.GoldMiner import GoldMinerEnv
from model.ShoppingEnv import ShoppingEnv
from agent.ShoppingAgent import ShoppingAgent
from agent.Qtention.Qtention import Qtention
from trainer.ShoppingBanditTrainer import ShoppingBanditTrainer
from trainer.QtentionTrainer import QtentionTrainer


def train_shopping_agent(
    mining_checkpoint: str,
    num_episodes: int = 10000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    levels: list = None,
    save_freq: int = 500,
    save_dir: str = 'checkpoints/shopping',
    resume_from: str = None,
    show: bool = False,
    fps: int = 60,
):
    """
    Train shopping agent
    
    Args:
        mining_checkpoint: Path to pre-trained mining agent checkpoint
        num_episodes: Số episodes để train
        epsilon_start: Starting epsilon for exploration
        epsilon_end: Minimum epsilon
        epsilon_decay: Epsilon decay rate
        levels: List levels để train (default: 1-9)
        save_freq: Save checkpoint mỗi N episodes
        save_dir: Directory để save checkpoints
        resume_from: Path to checkpoint để resume training
        show: Show game render trong quá trình train (default: False)
        fps: Frames per second khi show (default: 60)
    """
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if levels is None:
        levels = list(range(1, 10))  # 1-9
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("SHOPPING AGENT TRAINING")
    print("="*60)
    print(f"Mining checkpoint: {mining_checkpoint}")
    print(f"Levels: {levels}")
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    print(f"Save directory: {save_dir}")
    print("="*60 + "\n")
    
    # ==================== MINING SETUP ====================
    print("Setting up mining environment and agent...")
    
    # Create mining env
    render_mode = 'human' if show else None
    mining_env = GoldMinerEnv(
        render_mode=render_mode,
        max_steps=3600,
        levels=levels,
        use_generated_levels=True,
        c_dyna=0,
        c_step=0.0,
        c_pull=0.0,
        reward_scale=1.0,
        game_speed=fps / 60.0,  # Scale game speed theo FPS
    )
    print(f"✓ Mining env created (render={render_mode}, fps={fps if show else 'N/A'})")
    
    # Create mining agent
    mining_agent = Qtention(
        d_model=32,
        n_actions=50,
        nhead=8,
        n_layers=6,
        d_ff=48,
        dropout=0.1,
        max_items=30
    )
    print("✓ Mining agent (Qtention) created")
    
    # Create mining trainer
    mining_trainer = QtentionTrainer(
        env=mining_env,
        agent=mining_agent,
        lr=1e-4,
        gamma=1.0,
        epsilon_start=0.0,  # No exploration during shopping training
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=1,
        batch_size=1,
        target_update_freq=1,
        train_freq=1,
        num_planning=1,
        use_planning=False
    )
    
    # Load mining checkpoint
    print(f"Loading mining checkpoint: {mining_checkpoint}")
    mining_trainer.load_checkpoint(mining_checkpoint)
    print(f"✓ Mining agent loaded (steps: {mining_trainer.total_steps})")
    
    # ==================== SHOPPING SETUP ====================
    print("\nSetting up shopping environment and agent...")
    
    # Create shopping env
    shopping_env = ShoppingEnv(
        mining_env=mining_env,
        mining_trainer=mining_trainer,
        levels=levels,
    )
    print("✓ Shopping env created")
    
    # Create shopping agent
    shopping_agent = ShoppingAgent(
        num_levels=len(levels),
        num_actions=32,
    )
    
    # Resume if specified
    if resume_from:
        print(f"Resuming from: {resume_from}")
        shopping_agent.load(resume_from)
        print(f"✓ Resumed (total_episodes: {shopping_agent.total_episodes})")
    else:
        print("✓ Shopping agent created (new)")
    
    # Create shopping trainer
    shopping_trainer = ShoppingBanditTrainer(
        env=shopping_env,
        agent=shopping_agent,
    )
    print("✓ Shopping trainer created")
    
    # ==================== TRAINING ====================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    save_path = os.path.join(save_dir, f"shopping_agent_{timestamp}")
    
    episode_rewards = shopping_trainer.train(
        num_episodes=num_episodes,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        save_freq=save_freq,
        save_path=save_path,
    )
    
    # ==================== FINAL SAVE ====================
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    final_path = os.path.join(save_dir, f"shopping_agent_{timestamp}_final.npz")
    shopping_agent.save(final_path)
    print(f"✓ Final agent saved: {final_path}")
    
    # Statistics
    stats = shopping_agent.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Visit counts per level: {stats['visit_counts']}")
    print(f"  Mean Q per level: {[f'{q:.1f}' for q in stats['mean_q_per_level']]}")
    print(f"  Final epsilon: {stats['epsilon']:.4f}")
    
    # Recent performance
    if len(episode_rewards) >= 100:
        recent_avg = np.mean(episode_rewards[-100:])
        print(f"  Avg reward (last 100): {recent_avg:.1f}")
    
    print("\n" + "="*60)
    
    # Cleanup
    mining_env.close()
    
    return shopping_agent, episode_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Shopping Agent for Gold Miner')
    
    parser.add_argument('--mining-checkpoint', type=str, 
                        default='checkpoints/qtention/checkpoint_cycle_2000.pth',
                        help='Path to pre-trained mining agent checkpoint')
    
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes (default: 10000)')
    
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon (default: 1.0)')
    
    parser.add_argument('--epsilon-end', type=float, default=0.32,
                        help='Minimum epsilon (default: 0.32)')
    
    parser.add_argument('--epsilon-decay', type=float, default=0.99,
                        help='Epsilon decay rate (default: 0.99)')
    
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                        help='Levels to train on (default: 1-9)')
    
    parser.add_argument('--save-freq', type=int, default=500,
                        help='Save checkpoint every N episodes (default: 500)')
    
    parser.add_argument('--save-dir', type=str, default='checkpoints/shopping',
                        help='Directory to save checkpoints (default: checkpoints/shopping)')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (default: None)')
    
    parser.add_argument('--show', action='store_true',
                        help='Show game render during training (default: False)')
    
    parser.add_argument('--fps', type=int, default=60,
                        help='Frames per second when showing (default: 60)')
    
    args = parser.parse_args()
    
    train_shopping_agent(
        mining_checkpoint=args.mining_checkpoint,
        num_episodes=args.episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        levels=args.levels,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        resume_from=args.resume,
        show=args.show,
        fps=args.fps,
    )
