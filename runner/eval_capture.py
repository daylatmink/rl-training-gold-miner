"""
Evaluation script với chức năng capture level
Load checkpoint và render game, cho phép lưu initial state của episode dưới format levels.json
Nhấn SPACE để lưu level hiện tại
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pygame
import argparse
import json
import random
from datetime import datetime
from model.GoldMiner import GoldMinerEnv
from agent.Qtention.Qtention import Qtention
from agent.QCNN.QCNN import QCNN
from agent.QCnnRnn.QCnnRnn import QCnnRnn
from trainer.QtentionTrainer import QtentionTrainer
from trainer.QcnnTrainer import QcnnTrainer
from trainer.QCnnRnnTrainer import QCnnRnnTrainer
from agent.Qtention.Embedder import Embedder
from trainer.QtentionTrainer import angle_bins


def state_to_level_format(state: dict) -> dict:
    """
    Convert game state sang format levels.json
    
    Args:
        state: Game state dict từ _get_observation()
        
    Returns:
        dict: Level data theo format của levels.json
    """
    entities = []
    
    # Map từ item size (pixel) sang entity type
    gold_size_mapping = {
        30: 'MiniGold',
        70: 'NormalGold',
        90: 'NormalGoldPlus',
        150: 'BigGold',
    }
    
    rock_size_mapping = {
        30: 'MiniRock',
        60: 'NormalRock',
        100: 'BigRock',
    }
    
    for item in state['items']:
        item_type = item['type']
        item_size = item.get('size', None)
        item_subtype = item.get('subtype', None)
        
        # Determine entity type
        if item_type == 'Gold':
            entity_type = gold_size_mapping.get(item_size, 'NormalGold')
        elif item_type == 'Rock':
            entity_type = rock_size_mapping.get(item_size, 'NormalRock')
        elif item_type == 'Other':
            # Map subtype for Other items
            if item_subtype == 'Diamond':
                entity_type = 'Diamond'
            elif item_subtype == 'Skull':
                entity_type = 'Skull'
            elif item_subtype == 'Bone':
                entity_type = 'Bone'
            else:
                entity_type = 'Unknown'
        elif item_type == 'Mole':
            if item_subtype == 'MoleWithDiamond':
                entity_type = 'MoleWithDiamond'
            else:
                entity_type = 'Mole'
        elif item_type == 'QuestionBag':
            entity_type = 'QuestionBag'
        elif item_type == 'TNT':
            entity_type = 'TNT'
        elif item_type == 'Hoo':
            entity_type = 'Hoo'
        else:
            entity_type = item_type
        
        entity = {
            'type': entity_type,
            'pos': {
                'x': int(item['position']['x']),
                'y': int(item['position']['y'])
            }
        }
        
        # Thêm direction nếu có (cho Mole, QuestionBag)
        if 'direction' in item:
            entity['dir'] = 'Left' if item['direction'] < 0 else 'Right'
        
        entities.append(entity)
    
    level_data = {
        'type': 'LevelA',  # Default level type
        'entities': entities
    }
    
    return level_data


def save_level(level_data: dict, save_dir: str, level_name: str = None) -> str:
    """
    Lưu level data vào file JSON
    
    Args:
        level_data: Level data theo format levels.json
        save_dir: Thư mục lưu
        level_name: Tên level (optional, sẽ tự generate nếu không có)
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if level_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        level_name = f'captured_level_{timestamp}'
    
    # Tạo file riêng cho mỗi level
    file_path = os.path.join(save_dir, f'{level_name}.json')
    
    level_json = {level_name: level_data}
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(level_json, f, indent=4, ensure_ascii=False)
    
    return file_path


def load_levels_from_dir(test_dir: str) -> list:
    """
    Load tất cả level files từ một thư mục
    
    Args:
        test_dir: Đường dẫn thư mục chứa các file level JSON
        
    Returns:
        list: List of (filename, level_data) tuples
    """
    levels = []
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return levels
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(test_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # File format: {"level_name": {"type": "...", "entities": [...]}}
                    for level_name, level_data in data.items():
                        levels.append((level_name, level_data))
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    
    return levels


def evaluate_with_capture(checkpoint_path: str, num_episodes: int = 5, fps: int = 60, 
                          net: str = "attention", save_dir: str = "captured_levels", seed: int = None,
                          test_dir: str = None):
    """
    Load agent từ checkpoint và chơi game với khả năng capture level
    Nhấn SPACE để lưu initial state của episode hiện tại
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Số episodes để chơi
        fps: Frames per second (giới hạn FPS)
        net: Network architecture
        save_dir: Thư mục lưu captured levels
        seed: Random seed cho reproducibility (reset mỗi episode)
        test_dir: Thư mục chứa các level để test (nếu khác None sẽ eval các level này)
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment với render mode
    print("\nCreating environment...")
    env = GoldMinerEnv(
        render_mode='human',
        max_steps=3600,
        levels=10,
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0.0,
        reward_scale=10000.0,
        game_speed=fps / 60.0
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
    
    # Create trainer
    print("\nInitializing trainer...")
    if net == "attention":
        trainer = QtentionTrainer(
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
    else:
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
    
    # Ensure save directory exists (chỉ khi không phải test mode)
    if test_dir is None and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSave directory: {save_dir}")
    
    print("\n" + "="*60)
    print("Starting evaluation with capture...")
    
    # Load levels from test_dir if specified
    test_levels = None
    if test_dir is not None:
        test_levels = load_levels_from_dir(test_dir)
        if test_levels:
            print(f"Loaded {len(test_levels)} levels from {test_dir}")
            num_episodes = len(test_levels)  # Override num_episodes với số levels
        else:
            print(f"⚠️ No levels found in {test_dir}, using generated levels")
    
    print(f"Playing {num_episodes} episodes with {fps} FPS limit")
    if seed is not None:
        print(f"Random seed: {seed} (reset each episode for reproducibility)")
    print("Press SPACE to save current episode's initial state")
    print("Press ENTER to skip current episode")
    print("Press ESC or close window to stop")
    print("="*60 + "\n")
    
    episode_rewards = []
    captured_count = 0
    
    try:
        for episode in range(1, num_episodes + 1):
            print(f"\n{'='*60}")
            
            # Get level data if testing from directory
            current_level_data = None
            current_level_name = None
            if test_levels is not None and episode <= len(test_levels):
                current_level_name, current_level_data = test_levels[episode - 1]
                print(f"Episode {episode}/{num_episodes}: {current_level_name}")
            else:
                print(f"Episode {episode}/{num_episodes}")
            print('='*60)
             
            # Reset selective_rng với seed cố định cho mỗi episode
            # Chỉ ảnh hưởng đến selective greedy, không ảnh hưởng level generation
            if seed is not None:
                trainer.set_selective_seed(seed)
            
            # Reset environment và lưu initial state
            # Nếu có test level thì inject vào options
            reset_options = {'level_data': current_level_data} if current_level_data else None
            state, _ = env.reset(options=reset_options)
            initial_state = state.copy()
            initial_level_data = state_to_level_format(initial_state)
            level_saved = False
            
            # Manual evaluation loop (thay vì dùng trainer.evaluate)
            episode_reward = 0.0
            action_buffer = None
            reward_buffer = 0
            angle_decision = None
            done = False
            prev_total_points = 0.0
            miss_streak = 0
            prev_num_items = -1
            
            trainer.agent.eval()
            
            while True:
                # Check for SPACE key press
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and not level_saved and test_levels is None:
                            # Save initial state (chỉ khi không phải test mode)
                            level_name = f'episode_{episode}_{datetime.now().strftime("%H%M%S")}'
                            saved_path = save_level(initial_level_data, save_dir, level_name)
                            print(f"\n>>> LEVEL SAVED: {saved_path}")
                            level_saved = True
                            captured_count += 1
                        elif event.key == pygame.K_RETURN:
                            # Skip episode - không lưu, không tiếp tục
                            print(f"\n>>> EPISODE SKIPPED")
                            done = True
                        elif event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                    elif event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
                    next_state, reward, terminated, truncated, info = env.step(0)
                else:
                    if angle_decision is None or done:
                        if action_buffer is not None:
                            # Phát hiện TNT explosion
                            cur_total_points = sum(item['point'] for item in state['items'])
                            if reward_buffer == 0 and cur_total_points < prev_total_points:
                                lost_points = prev_total_points - cur_total_points
                                tnt_penalty = -env.c_tnt * lost_points / env.reward_scale
                                reward_buffer += tnt_penalty
                            prev_total_points = cur_total_points
                            
                            # Track miss_streak
                            cur_num_items = len(state['items'])
                            if cur_num_items == prev_num_items:
                                miss_streak += 1
                            else:
                                miss_streak = 0
                            prev_num_items = cur_num_items
                            
                            episode_reward += reward_buffer
                            reward_buffer = 0
                            action_buffer = None
                        if done:
                            break
                        
                        # Greedy action selection
                        action_buffer, used_model = trainer.select_action(state, miss_streak=miss_streak, training=False)
                        angle_decision = angle_bins[action_buffer]
                    
                    current_angle = state['rope_state']['direction']
                    if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                        next_state, reward, terminated, truncated, info = env.step(1)
                        angle_decision = None
                    else:
                        next_state, reward, terminated, truncated, info = env.step(0)
                
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
            
            trainer.agent.train()
            episode_rewards.append(episode_reward)
            
            print(f"\n{'='*60}")
            print(f"Episode {episode} completed!")
            print(f"  Reward: {episode_reward:.3f}")
            if level_saved:
                print(f"  Level was captured!")
            print('='*60)
    
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
        print(f"Levels captured: {captured_count}")
        print(f"Save directory: {save_dir}")
        print("="*60)
    
    print("\n✓ Evaluation completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained agent with level capture')
    parser.add_argument('--checkpoint', type=str, 
                        default='C:\\Users\\User\\Documents\\code\\rl-training-gold-miner\\checkpoints\\qtention\\checkpoint_cycle_2000.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to play (default: 100)')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit (default: 60)')
    parser.add_argument('--net', type=str, default='attention', choices=['attention', 'cnn', 'cnn_rnn'],
                        help='Network architecture (default: attention)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory to save captured levels (default: captured_levels)')
    parser.add_argument('--seed', type=int, default=49,
                        help='Random seed for reproducibility (reset each episode, default: 49)')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Directory containing saved levels to test (if set, will eval these levels instead of generating new ones)')
    
    args = parser.parse_args()
    
    evaluate_with_capture(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        fps=args.fps,
        net=args.net,
        save_dir=args.save_dir,
        seed=args.seed,
        test_dir=args.test_dir
    )
