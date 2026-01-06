"""
Evaluation script để xem agent chơi Gold Miner
Load checkpoint và render game với visualization (tận dụng DQNTrainer.evaluate)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pygame
import argparse
from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Qtention import Qtention
from agent.QCNN.QCNN import QCNN
from agent.QCnnRnn.QCnnRnn import QCnnRnn
from trainer.QtentionTrainer import QtentionTrainer
from trainer.QcnnTrainer import QcnnTrainer
from trainer.QCnnRnnTrainer import QCnnRnnTrainer


def evaluate_agent_with_render(checkpoint_path: str, num_episodes: int = 5, fps: int = 60, net: str = "attention", seed: int = None):
    """
    Load agent từ checkpoint và chơi game với visualization
    Tận dụng DQNTrainer để có logic giống hệt training
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Số episodes để chơi
        fps: Frames per second (giới hạn FPS)
        seed: Random seed cho reproducibility (reset mỗi episode)
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment với render mode
    print("\nCreating environment...")
    env = GoldMinerEnv(
        render_mode='human',  # Hiển thị game
        max_steps=3600,       # 60 giây * 60 FPS
        levels=10,
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0.0,
        reward_scale=10000.0,
        game_speed=fps / 60.0  # Scale game speed theo FPS
    )
    print("✓ Environment created")
    
    # Create agent
    print(f"\nCreating agent ({net})...")
    
    if net == "attention":
        agent = Qtention(
            d_model=32,
            n_actions=50,
            nhead=8,
            n_layers=6,
            d_ff=48,
            dropout=0.1,
            max_items=30
        )
    elif net == "cnn":
        agent = QCNN(
            d_model=24,
            n_actions=50,
            d_hidden=24,
            dropout=0.2
        )
    elif net == "cnn_rnn":
        agent = QCnnRnn(
            d_model=24,
            n_actions=50,
            d_hidden=48,
            dropout=0.1
        )
    else:
        raise ValueError(f"Invalid network type: {net}. Choose 'attention', 'cnn', or 'cnn_rnn'")
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"✓ Agent created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer (không train, chỉ dùng evaluate method)
    print("\nInitializing trainer...")
    if net == "attention":
        trainer = QtentionTrainer(
            env=env,
            agent=agent,
            lr=1e-4,
            gamma=1.0,
            epsilon_start=0.0,  # No exploration cho eval
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=1,  # Không cần buffer
            batch_size=1,
            target_update_freq=1,
            train_freq=1,
            num_planning=1,
            use_planning=False
        )
    elif net == "cnn":
        trainer = QcnnTrainer(
            env=env,
            agent=agent,
            lr=1e-4,
            gamma=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=1,
            batch_size=1,
            target_update_freq=1,
            train_freq=1,
            num_planning=1,
            use_planning=False
        )
    else:  # cnn_rnn
        trainer = QCnnRnnTrainer(
            env=env,
            agent=agent,
            lr=1e-4,
            gamma=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=1,
            batch_size=1,
            target_update_freq=1,
            train_freq=1,
            num_planning=1,
            use_planning=False
        )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)
    print("✓ Checkpoint loaded")
    
    if trainer.total_steps > 0:
        print(f"  Training steps: {trainer.total_steps}")
    print(f"  Epsilon: {trainer.epsilon:.4f}")
    
    # Clock for FPS control
    clock = pygame.time.Clock()
    
    print("\n" + "="*60)
    print("Starting evaluation...")
    print(f"Playing {num_episodes} episodes with {fps} FPS limit")
    if seed is not None:
        print(f"Random seed: {seed} (reset each episode for reproducibility)")
    print("Press ESC or close window to stop")
    print("="*60 + "\n")
    
    episode_rewards = []
    
    try:
        for episode in range(1, num_episodes + 1):
            print(f"\n{'='*60}")
            print(f"Episode {episode}/{num_episodes}")
            print('='*60)
            
            # Reset selective_rng với seed cố định cho mỗi episode
            # Chỉ ảnh hưởng đến selective greedy, không ảnh hưởng level generation
            if seed is not None:
                trainer.set_selective_seed(seed)
            
            # Sử dụng evaluate() method từ trainer (chỉ 1 episode)
            avg_reward = trainer.evaluate(num_episodes=1)
            episode_rewards.append(avg_reward)
            
            print(f"\n{'='*60}")
            print(f"Episode {episode} completed!")
            print(f"  Reward: {avg_reward:.3f}")
            print('='*60)
            
            # Note: FPS limiting được handle bởi game_speed parameter
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        env.close()
    
    # Print summary
    if episode_rewards:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes played: {len(episode_rewards)}")
        print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
        print(f"Best reward: {max(episode_rewards):.3f}")
        print(f"Worst reward: {min(episode_rewards):.3f}")
        print("="*60)
    
    print("\n✓ Evaluation completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent for Gold Miner')
    parser.add_argument('--checkpoint', type=str, default='C:\\Users\\User\\Documents\\code\\rl-training-gold-miner\\checkpoints\\qtention\\checkpoint_cycle_2000.pth',
                        help='Path to checkpoint file (default: None)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 3)')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit (default: 60)')
    parser.add_argument('--net', type=str, default='attention', choices=['attention', 'cnn', 'cnn_rnn'],
                        help='Network architecture: "attention" (Qtention), "cnn" (QCNN), or "cnn_rnn" (QCnnRnn) (default: attention)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (reset each episode, default: None)')
    
    args = parser.parse_args()
    
    evaluate_agent_with_render(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        fps=args.fps,
        net=args.net,
        seed=args.seed
    )
