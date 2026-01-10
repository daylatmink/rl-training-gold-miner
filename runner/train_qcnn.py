"""
Script để chạy Double QCNN Trainer
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from model.GoldMiner import GoldMinerEnv
from agent.QCNN.QCNN import QCNN
from trainer.DoubleQCNNTrainer import DoubleQCNNTrainer


def main():
    parser = argparse.ArgumentParser(description='Train QCNN network with Double DQN')
    
    # Training strategy
    parser.add_argument('--n-cycles', type=int, default=2000,
                        help='Number of training cycles (default: 2000)')
    parser.add_argument('--n-updates', type=int, default=3,
                        help='Number of Q-updates per cycle (default: 3)')
    parser.add_argument('--m-episodes', type=int, default=2,
                        help='Number of episodes to collect per cycle (default: 2)')
    
    # Buffer
    parser.add_argument('--warmup-buffer', type=str, default='buffers/warmup_buffer_QCNN.pkl',
                        help='Path to warmup buffer (default: buffers/warmup_buffer_QCNN.pkl)')
    parser.add_argument('--buffer-size', type=int, default=20000,
                        help='Replay buffer capacity (default: 20000)')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor (default: 0.9)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Target network update frequency (default: 100)')
    
    # Epsilon
    parser.add_argument('--epsilon-start', type=float, default=0.9,
                        help='Starting epsilon (default: 0.9)')
    parser.add_argument('--epsilon-end', type=float, default=0.5,
                        help='Ending epsilon (default: 0.5)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate (default: 0.995)')
    
    # Environment
    parser.add_argument('--show', action='store_true',
                        help='Show game window during training')
    parser.add_argument('--game-speed', type=int, default=1,
                        help='Game speed multiplier (default: 1)')
    
    # Saving
    parser.add_argument('--save-dir', type=str, default='checkpoints/QCNN',
                        help='Directory to save checkpoints (default: checkpoints/QCNN)')
    parser.add_argument('--log-file', type=str, default='training_QCNN.log',
                        help='Log file path (default: training_QCNN.log)')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("Double QCNN Training Setup")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = GoldMinerEnv(
        render_mode='human' if args.show else None,
        max_steps=3600,
        levels=list(range(1, 11)),
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0,
        reward_scale=10000.0,
        game_speed=args.game_speed
    )
    print("✓ Environment created")
    
    # Create agent
    print("\n[2/4] Creating QCNN agent...")
    agent = QCNN()
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"✓ Agent created: {total_params:,} parameters")
    # print(f"  d_model={args.d_model}, d_ff={args.d_ff}, nhead={args.nhead}, n_layers={args.n_layers}")
    
    # Create trainer
    print("\n[3/4] Creating trainer...")
    trainer = DoubleQCNNTrainer(
        env=env,
        agent=agent,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        max_items=30,
        device=device
    )
    print("✓ Trainer created")
    
    # Resume from checkpoint
    if args.resume:
        print(f"\n[3.5/4] Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n[4/4] Starting training...")
    print(f"  Strategy: {args.n_updates} updates → {args.m_episodes} episodes → repeat")
    print(f"  Cycles: {args.n_cycles}")
    print(f"  Warmup buffer: {args.warmup_buffer}")
    print(f"  Save dir: {args.save_dir}")
    
    try:
        trainer.train(
            n_cycles=args.n_cycles,
            n_updates_per_cycle=args.n_updates,
            m_episodes_per_cycle=args.m_episodes,
            warmup_buffer_path=args.warmup_buffer,
            save_dir=args.save_dir,
            log_file=args.log_file
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        # Save checkpoint on interrupt
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint_interrupted.pth')
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    finally:
        env.close()
        print("\n✓ Environment closed")


if __name__ == '__main__':
    main()
