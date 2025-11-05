"""
Main script để train DQN agent cho Gold Miner
"""

from model.GoldMiner import GoldMinerEnv
from agent.Qtention import Qtention
from train_dqn import DQNTrainer


def main_train(headless: bool = False, checkpoint: str = None):
    """
    Main training function
    
    Args:
        headless: Nếu True, train không mở cửa sổ pygame (nhanh hơn)
                  Nếu False, train với cửa sổ hiển thị
        checkpoint: Path to checkpoint file to resume training
    """
    # Hyperparameters
    config = {
        'num_episodes': 1000,
        'lr': 3e-4,
        'gamma': 1,
        'epsilon_start': 0.1,    # Tăng exploration ban đầu để học cả 2 actions
        'epsilon_end': 0.001,     # Giữ một chút exploration
        'epsilon_decay': 0.995,  # Decay chậm hơn: 0.3 -> 0.01 trong ~500 episodes
        'buffer_size': 4000,
        'batch_size': 128,
        'target_update_freq': 1000,
        'train_freq': 1,         # Train mỗi bao nhiêu steps
        'num_planning': 4,       # Số lần quét qua toàn bộ buffer mỗi lần train (planning approach)
        'save_freq': 1,
        'eval_freq': 200,
        'eval_episodes': 5,
        'headless': headless,     # Headless mode
    }
    
    print("="*60)
    print("Gold Miner DQN Training")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    render_mode = 'human' if not headless else None
    print(f"  Render mode: {render_mode} ({'with display' if not headless else 'headless - no window'})")
    env = GoldMinerEnv(
        render_mode=render_mode,  # None = headless (không mở cửa sổ), 'human' = hiển thị
        max_steps=3600,    # 60 giây * 60 FPS
        level=1,
        use_generated_levels=True,
        c_dyna=10,       # Cost của dynamite
        c_step=0.0,        # Step cost (0 = không dùng)
        c_pull=0.3,        # Penalty khi đang kéo (0 = không dùng)
        reward_scale=1000.0,  # Scale reward xuống 1000 lần
        game_speed=1       # Giữ 1x để physics chính xác, headless đã đủ nhanh
    )
    print("✓ Environment created")
    
    # Create agent
    print("\n[2/4] Creating agent...")
    agent = Qtention(
        d_model=20,
        n_actions=2,
        nhead=4,
        n_layers=2,
        d_ff=24,
        dropout=0.01,
        max_items=30
    )
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"✓ Agent created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n[3/4] Creating trainer...")
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        lr=config['lr'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq'],
        train_freq=config['train_freq'],
        num_planning=config['num_planning']
    )
    print(f"✓ Trainer created on device: {trainer.device}")
    
    # Load checkpoint if provided
    if checkpoint:
        print(f"\n[3.5/4] Loading checkpoint...")
        print(f"  Checkpoint: {checkpoint}")
        trainer.load_checkpoint(checkpoint)
        print(f"✓ Checkpoint loaded")
        print(f"  Resuming from episode {len(trainer.episode_rewards)+1}")
        print(f"  Total steps: {trainer.total_steps}")
        print(f"  Current epsilon: {trainer.epsilon:.4f}")
    
    # Start training
    print("\n[4/4] Starting training...")
    print("="*60)
    
    try:
        trainer.train(
            num_episodes=config['num_episodes'],
            save_freq=config['save_freq'],
            eval_freq=config['eval_freq'],
            eval_episodes=config['eval_episodes']
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
        trainer.save_checkpoint('checkpoints/interrupted_model.pt')
        trainer.save_training_log('training_log_interrupted.json')
        print("✓ Progress saved")
    finally:
        env.close()
        print("\n✓ Environment closed")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for Gold Miner')
    parser.add_argument('--show', action='store_true', 
                        help='Show pygame window during training (default: headless mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training (default: checkpoints/checkpoint.pt)')
    args = parser.parse_args()
    
    # Nếu dùng --show thì headless=False (hiển thị), ngược lại headless=True (không hiển thị)
    main_train(headless=not args.show, checkpoint=args.checkpoint)
    # main_train(headless=args.show)
