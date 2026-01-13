import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.GoldMiner import GoldMinerEnv
import argparse
from random import random
import numpy as np
import pickle
from runner.full_run import RandomMiningWrapper
import torch
from agent.Qtention.Qtention import Qtention
from trainer.QtentionTrainer import QtentionTrainer, angle_bins
from agent.QCNN.QCNN import QCNN
from trainer.DoubleQCNNTrainer import DoubleQCNNTrainer as QcnnTrainer
from define import (set_level, set_score, set_dynamite_count, get_dynamite_count, 
                    get_score, reset_game_state, get_level, set_goal, get_goal, goalAddOn)
from tqdm import tqdm


def load_mining_agent(checkpoint_path: str, device: str, env: GoldMinerEnv, use_random: bool = False, net = 'attention'):
    """Load trained mining agent with trainer wrapper or return random agent"""
    if use_random:
        print("Using random mining agent")
        return RandomMiningWrapper(env)
    
    print(f"Loading mining agent from: {checkpoint_path}")
    if net == 'attention':
        agent = Qtention(
            d_model=32,
            n_actions=50,
            nhead=8,
            n_layers=6,
            d_ff=48,
            dropout=0.1,
            max_items=30
        ).to(device)
        
        # Create trainer (for evaluate logic)
        trainer = QtentionTrainer(
            env=env,
            agent=agent,
            lr=1e-4,
            gamma=1.0,
            epsilon_start=0.0,  # No exploration for eval
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
        agent = QCNN().to(device)
        
        # Create trainer (for evaluate logic)
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
            target_update_freq=1
        )
    
    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    trainer.agent.eval()
    
    print(f"✓ Mining agent loaded")
    return trainer

def run_mining_phase(
    trainer,  # QtentionTrainer or RandomMiningWrapper
    level: int,
    current_dynamite: int = 0,  # Số dynamite hiện có
):
    """
    Chạy mining phase cho một level using trainer's evaluate logic
    
    Args:
        trainer: Mining agent trainer
        level: Level number
        current_dynamite: Số dynamite hiện có
    """
    # Set seed for this level if provided
    
    # Set global level state to ensure correct level is used
    set_level(level)
    
    # Lưu dynamite trước khi reset (vì reset_game_state() sẽ xóa dynamite)
    # Reset env to specific level with seed
    state, info = trainer.env.reset(options={'level': level})
    
    # Khôi phục dynamite
    set_dynamite_count(current_dynamite)
    # Cũng cập nhật rope.have_TNT trong game_scene
    if trainer.env.game_scene is not None:
        trainer.env.game_scene.rope.have_TNT = current_dynamite
    
    # Cập nhật goal theo level: goal = 650 + (level - 1) * goalAddOn
    level_goal = 650 + (level - 1) * goalAddOn
    set_goal(level_goal)
    
    episode_reward = 0.0
    episode_steps = 0
    action_buffer = None
    reward_buffer = 0
    angle_decision = None
    done = False
    miss_streak = 0
    prev_total_points = 0
    prev_num_items = -1
    res = {
        'other': 0,
        'rock': 0,
        'diamond': 0
    }
    
    def update(score):
        if score >= 600:
            res['diamond'] += 600
            res['other'] += score - 600
        elif score in [11, 20]:
            res['rock'] += score
        else:
            res['other'] += score
    
    # Main episode loop (similar to trainer.evaluate)
    while True:
        # Auto-use dynamite khi đang kéo đá lớn (BigRock)
        rope_state = state['rope_state']
        if (rope_state['state'] == 'retracting' and 
            rope_state['has_item'] and 
            rope_state['item_type'] == 'Rock' and
            rope_state['tnt_count'] > 0 and
            not rope_state.get('is_use_tnt', False)):
            # Kiểm tra xem có phải đá lớn không (dựa vào weight hoặc point)
            # Rock lớn có weight cao (base_weight * 3, với size >= 60 thì base_weight >= 2)
            if rope_state['weight'] >= 6:  # Đá lớn: size >= 60 → weight = (60/30) * 3 = 6
                # Tự động dùng dynamite
                next_state, reward, terminated, truncated, info = trainer.env.step(1)
                done = terminated or truncated
                reward_buffer += reward
                state = next_state
                continue
        
        if not done and (state['rope_state']['state'] in ['expanding', 'retracting'] or state['rope_state']['timer'] > 0):
            next_state, reward, terminated, truncated, info = trainer.env.step(0)  # No-op action
        else:
            if angle_decision is None or done:
                if action_buffer is not None:
                    # Detect TNT explosion
                    cur_total_points = sum(item['point'] for item in state['items'])
                    if reward_buffer == 0 and cur_total_points < prev_total_points:
                        lost_points = prev_total_points - cur_total_points
                        tnt_penalty = -trainer.env.c_tnt * lost_points / trainer.env.reward_scale
                        reward_buffer += tnt_penalty
                    prev_total_points = cur_total_points
                    
                    # Track miss_streak
                    cur_num_items = len(state['items'])
                    if cur_num_items == prev_num_items:
                        miss_streak += 1
                    else:
                        miss_streak = 0
                    prev_num_items = cur_num_items
                    
                    update(reward_buffer)
                    episode_reward += reward_buffer
                    episode_steps += 1
                    reward_buffer = 0
                    action_buffer = None
                if done:
                    break
                action_buffer, used_model = trainer.select_action(state, miss_streak=miss_streak, training=False)  # Greedy
                angle_decision = angle_bins[action_buffer]
            
            current_angle = state['rope_state']['direction']
            if angle_decision is not None and angle_decision[0] <= current_angle and current_angle < angle_decision[1]:
                next_state, reward, terminated, truncated, info = trainer.env.step(1)  # Fire action
                angle_decision = None
            else:
                next_state, reward, terminated, truncated, info = trainer.env.step(0)  # No-op action
    
        done = terminated or truncated
        reward_buffer += reward
        state = next_state
    
    return res


def main():
    parser = argparse.ArgumentParser(description='Run full Gold Miner simulation')
    
    parser.add_argument('--num-runs-per-level', type=int, default=5,
                        help='Number of runs to execute per level (default: 5)')
    parser.add_argument('--checkpoint', type=str, default='C:\\Users\\User\\Documents\\code\\rl-training-gold-miner\\checkpoints\\qtention\\checkpoint_cycle_2000.pth',
                        help='Path to mining agent checkpoint (default: None for random guessing)')
    parser.add_argument('--save-results', type=str, required=True,
                        help='Path to save results (default: None)')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit (default: 60)')
    parser.add_argument('--show', action='store_true',
                        help='Show game window during simulation')
    parser.add_argument('--net', type=str, default='attention',
                        help="Type of shopping agent network: 'attention', 'cnn', or 'cnn_rnn' (default: 'attention')")
    
    args = parser.parse_args()
    
    # Detect random mode from checkpoint paths
    use_random_mining = (args.checkpoint is None)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    print(f"\n{'='*80}")
    print(f"FULL SIMULATION - {args.num_runs_per_level} RUNS x {10} LEVELS")
    print(f"{'='*80}")
    print(f"Mining agent: {'Random' if use_random_mining else args.checkpoint}")
    print(f"{'='*80}\n")
    
    # Create mining environment first
    print("Creating mining environment...")
    render_mode = 'human' if args.show else None
    mining_env = GoldMinerEnv(
        render_mode=render_mode,
        max_steps=3600,
        levels=list(range(1, 11)),  # List [1, 2, ..., num_levels] thay vì int
        use_generated_levels=True,
        c_dyna=0,
        c_step=0.0,
        c_pull=0.0,
        reward_scale=1.0,
        game_speed=args.fps / 60.0,  # Fast game speed
    )
    if args.show:
        print(f"✓ Mining environment created (render_mode=human, fps={args.fps})\n")
    else:
        print("✓ Mining environment created (no render for speed)\n")
    
    # Load mining agent (with trainer wrapper or random)
    mining_trainer = load_mining_agent(
        args.checkpoint, 
        device, 
        mining_env, 
        use_random=use_random_mining,
        net = args.net
    )
    
    # Run simulations
    all_results = [[] for x in range(10)]  # Lưu kết quả cho mỗi level
    
    for level in range(1, 11):
        for run_idx in tqdm(range(1, args.num_runs_per_level + 1)):
            
            result = run_mining_phase(
                trainer=mining_trainer,
                level=level,
                current_dynamite=0  # Bắt đầu với 0 dynamite
            )
            all_results[level-1].append(result)
    
    # Save results to file
    print(f"\nSaving results to: {args.save_results} ...")
    pickle.dump(all_results, open(args.save_results, 'wb'))
    print("✓ Results saved successfully!")    
    
    print("✓ All runs completed!\n")


if __name__ == '__main__':
    main()
