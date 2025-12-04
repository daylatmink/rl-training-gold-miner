"""
Main script để train DQN agent cho Gold Miner
"""

from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Qtention import Qtention
from agent.QCNN.QCNN import QCNN
from agent.QCnnRnn.QCnnRnn import QCnnRnn
from trainer.QCnnRnnTrainer import QCnnRnnTrainer
from trainer.DoubleQCnnRnnTrainer import DoubleQCnnRnnTrainer
from trainer.QtentionTrainer import QtentionTrainer
from trainer.QcnnTrainer import QcnnTrainer


def main_train(headless: bool = False, checkpoint: str = None, net: str = "attention", warmup_path: str = None, strategy: str = "Q", explore_strategy: str = "Exponentially"):
    """
    Main training function
    
    Args:
        headless: Nếu True, train không mở cửa sổ pygame (nhanh hơn)
                  Nếu False, train với cửa sổ hiển thị
        checkpoint: Path to checkpoint file to resume training
        net: Network architecture - "attention" (Qtention), "cnn" (QCNN), hoặc "cnn_rnn" (QCnnRnn)
        strategy: Training strategy - "Q" (DQN) hoặc "DoubleQ" (Double DQN), chỉ áp dụng cho cnn_rnn
    """
    # Hyperparameters
    # config = {
    #     'num_episodes': 15000,
    #     'lr': 3e-4,
    #     'gamma': 0.99,
    #     'epsilon_start': 0.4,    # Tăng exploration ban đầu để học cả 2 actions
    #     'epsilon_end': 0.05,     # Giữ một chút exploration
    #     'epsilon_decay': 0.8,  # Decay chậm hơn: 0.3 -> 0.01 trong ~500 episodes
    #     'buffer_size': 10000,
    #     'batch_size': 64,
    #     'target_update_freq': 20,
    #     'train_freq': 1,         # Train mỗi 1 step (tăng overhead, giảm tốc độ)
    #     'num_planning': 2,       # Số lần quét buffer (planning) hoặc số batches (standard)
    #     'use_planning': False,    # True: planning approach, False: standard DQN
    #     'warmup_steps': 1000,    # Số steps warmup với random actions trước khi train
    #     'save_freq': 100,
    #     'eval_freq': 500,         # Evaluate mỗi 50 episodes thay vì 200
    #     'eval_episodes': 5,      # Chỉ 5 episodes cho mỗi lần eval (nhanh hơn)
    #     'levels': list(range(1, 11)),      # List các levels, mỗi episode sẽ sample ngẫu nhiên
    #     'headless': headless,     # Headless mode
    # }
    
    config = {
        'num_episodes': 1000,
        'lr': 3e-4,
        'gamma': 0.9,
        'epsilon_start': 0.6,    # Tăng exploration ban đầu để học cả 2 actions
        'epsilon_end': 0.1,     # Giữ một chút exploration
        'epsilon_decay': 0.0000333,  # Decay chậm hơn: 0.3 -> 0.01 trong ~500 episodes
        'buffer_size': 500,
        'batch_size': 64,
        'target_update_freq': 20,
        'train_freq': 1,         # Train mỗi 1 step (tăng overhead, giảm tốc độ)
        'num_planning': 2,       # Số lần quét buffer (planning) hoặc số batches (standard)
        'use_planning': False,    # True: planning approach, False: standard DQN
        'warmup_steps': 500,    # Số steps warmup với random actions trước khi train
        'save_freq': 100,
        'eval_freq': 500,         # Evaluate mỗi 50 episodes thay vì 200
        'eval_episodes': 5,      # Chỉ 5 episodes cho mỗi lần eval (nhanh hơn)
        'levels': list(range(1, 11)),      # List các levels, mỗi episode sẽ sample ngẫu nhiên
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
        max_steps=3600,        # 60 giây * 60 FPS
        levels=config['levels'],  # List các levels để sample ngẫu nhiên
        use_generated_levels=True,
        c_dyna=10,       # Cost của dynamite
        c_step=0.0,        # Step cost (0 = không dùng)
        c_pull=0,        # Penalty khi đang kéo (0 = không dùng)
        c_miss=10.0,       # Penalty khi miss: miss lần đầu -a, miss lần 2 -2a (tổng -3a), ...
        c_tnt=0.7,         # Penalty khi kéo trúng TNT: phạt = c_tnt * tổng giá trị items bị phá hủy
        reward_scale=10000.0,  # Scale reward xuống 10000 lần
        game_speed=1       # Giữ 1x để physics chính xác, headless đã đủ nhanh
    )
    print("✓ Environment created")
    
    # Create agent
    print("\n[2/4] Creating agent...")
    print(f"  Network architecture: {net}")
    
    if net == "attention":
        agent = Qtention(
            d_model=20,
            n_actions=50,
            nhead=4,
            n_layers=3,
            d_ff=24,
            dropout=0.1,
            max_items=20
        )
    elif net == "cnn":
        agent = QCNN(
            d_model=24,
            n_actions=50,
            d_hidden=24,
            dropout=0.2
        )
    elif net == 'cnn_rnn':
        agent = QCnnRnn(
            d_model=24,
            n_actions=50,
            d_hidden=48,
            dropout=0.2,
            max_streak=15,
            num_layers=1
        )
    else:
        raise ValueError(f"Invalid network type: {net}. Choose 'attention' or 'cnn'")
    
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"✓ Agent created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n[3/4] Creating trainer...")
    if net == "attention":
        trainer = QtentionTrainer(
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
            num_planning=config['num_planning'],
            use_planning=config['use_planning']
        )
    elif net == "cnn":
        trainer = QcnnTrainer(
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
            num_planning=config['num_planning'],
            use_planning=config['use_planning'],
            warmup_steps=config['warmup_steps']
        )
    elif net == 'cnn_rnn':
        if strategy == "Q":
            trainer = QCnnRnnTrainer(
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
                num_planning=config['num_planning'],
                use_planning=config['use_planning'],
                warmup_steps=config['warmup_steps']
            )
        elif strategy == "DoubleQ":
            trainer = DoubleQCnnRnnTrainer(
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
                num_planning=config['num_planning'],
                use_planning=config['use_planning'],
                warmup_steps=config['warmup_steps']
            )
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Choose 'Q' or 'DoubleQ'")
    else:
        raise ValueError(f"Invalid network type for trainer: {net}. Choose 'attention', 'cnn', or 'cnn_rnn'")
    
    print(f"✓ Trainer created on device: {trainer.device}")
    print(f"  Training mode: {'Planning' if config['use_planning'] else 'Standard DQN'}")
    if net == 'cnn_rnn':
        print(f"  Strategy: {strategy} ({'Double DQN' if strategy == 'DoubleQ' else 'DQN'})")
    
    # Load checkpoint if provided (load trước để có replay buffer từ checkpoint)
    if checkpoint:
        print(f"\n[3.5/4] Loading checkpoint...")
        print(f"  Checkpoint: {checkpoint}")
        trainer.load_checkpoint(checkpoint)
        print(f"✓ Checkpoint loaded")
        print(f"  Resuming from episode {len(trainer.episode_rewards)+1}")
        print(f"  Total steps: {trainer.total_steps}")
        print(f"  Current epsilon: {trainer.epsilon:.4f}")
    
    # Load warmup buffer if provided (chỉ load nếu checkpoint không có buffer hoặc buffer rỗng)
    if warmup_path and len(trainer.replay_buffer) == 0:
        print(f"\n  Note: Replay buffer is empty, loading warmup buffer...")
        trainer.warmup_buffer(warmup_path)
    
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
                        help='Path to checkpoint file to resume training')
    parser.add_argument('--warmup-path', type=str, default=None,
                        help='Path to warmup buffer file to preload replay buffer')
    parser.add_argument('--net', type=str, default='attention', choices=['attention', 'cnn', 'cnn_rnn'],
                        help='Network architecture: attention (Qtention) or cnn (QCNN) or cnn_rnn (QCnnRnn) (default: attention)')
    parser.add_argument('--strategy', type=str, default='Q', choices=['Q', 'DoubleQ'],
                        help='Training strategy for cnn_rnn: Q (DQN) or DoubleQ (Double DQN) (default: Q)')
    parser.add_argument('--explore-strategy', type=str, default='Exponentially', choices=['Linearly', 'Exponentially'],
                        help='Epsilon decay strategy: Linearly or Exponentially (default: Exponentially)')
    args = parser.parse_args()
    
    # Nếu dùng --show thì headless=False (hiển thị), ngược lại headless=True (không hiển thị)
    main_train(headless=not args.show, checkpoint=args.checkpoint, net=args.net, warmup_path=args.warmup_path, strategy=args.strategy, explore_strategy=args.explore_strategy)
    # main_train(headless=False, net='cnn_rnn', warmup_path=r"C:\Users\User\Documents\code\rl-training-gold-miner\warmup_buffer_rnn.pkl", checkpoint=r'C:\Users\User\Documents\code\rl-training-gold-miner\checkpoints\cnn_rnn_ckpt.pt')