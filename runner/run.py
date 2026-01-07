"""
Full Simulation: Mining -> Shopping -> Mining -> ... qua 10 levels

Chạy nhiều runs và log kết quả trung bình
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json

from model.GoldMiner import GoldMinerEnv
from model.ShoppingEnv import ShoppingEnv, generate_random_shop, get_valid_actions, decode_action, calculate_cost
from agent.Qtention.Qtention import Qtention
from agent.ShoppingAgent import ShoppingAgent
from trainer.QtentionTrainer import QtentionTrainer, angle_bins
from define import set_level, set_score, set_dynamite_count, get_dynamite_count, get_score, reset_game_state, get_level
import random


class RandomMiningWrapper:
    """Wrapper for random mining actions"""
    def __init__(self, env):
        self.env = env
        self.agent = self  # Self-reference for compatibility
    
    def select_action(self, state, miss_streak=0, training=False):
        """Select random action"""
        action = random.randint(0, 49)  # 50 actions for mining
        return action, 'random'


class RandomShoppingAgent:
    """Random shopping agent"""
    def __init__(self):
        pass
    
    def get_action(self, obs, shop_state=None, money=None):
        """Select random action"""
        return random.randint(0, 31)  # 32 actions for shopping
    
    def set_eval_mode(self):
        pass


def load_mining_agent(checkpoint_path: str, device: str, env: GoldMinerEnv, use_random: bool = False):
    """Load trained mining agent with trainer wrapper or return random agent"""
    if use_random:
        print("Using random mining agent")
        return RandomMiningWrapper(env)
    
    print(f"Loading mining agent from: {checkpoint_path}")
    
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
    
    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    trainer.agent.eval()
    
    print(f"✓ Mining agent loaded")
    return trainer


def load_shopping_agent(checkpoint_path: str, use_random: bool = False):
    """Load trained shopping agent or return random agent"""
    if use_random:
        print("Using random shopping agent")
        return RandomShoppingAgent()
    
    print(f"Loading shopping agent from: {checkpoint_path}")
    
    agent = ShoppingAgent(num_levels=9, num_actions=32)
    agent.load(checkpoint_path)
    agent.set_eval_mode()
    
    print(f"✓ Shopping agent loaded")
    return agent


def run_mining_phase(
    trainer,  # QtentionTrainer or RandomMiningWrapper
    level: int,
    seed: int = None,
) -> Dict:
    """
    Chạy mining phase cho một level using trainer's evaluate logic
    
    Args:
        trainer: Mining agent trainer
        level: Level number
        seed: Random seed for reproducibility
    
    Returns:
        Dict với 'score', 'success', 'goal', 'reward', 'steps'
    """
    # Set seed for this level if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Set global level state to ensure correct level is used
    set_level(level)
    
    # Reset env to specific level with seed
    state, info = trainer.env.reset(seed=seed, options={'level': level})
    
    episode_reward = 0.0
    episode_steps = 0
    action_buffer = None
    reward_buffer = 0
    angle_decision = None
    done = False
    prev_total_points = 0.0
    miss_streak = 0
    prev_num_items = -1
    
    # Main episode loop (similar to trainer.evaluate)
    while True:
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
    
    # Extract results from final info
    # Get actual gold earned from define.get_score()
    gold_earned = get_score()
    
    return {
        'gold_earned': gold_earned,
        'reward': episode_reward,
        'steps': episode_steps,
        'level': level
    }


def run_shopping_phase(
    env,  # ShoppingEnv or None (if random)
    agent,  # ShoppingAgent or RandomShoppingAgent
    level: int,
    money: int,
    seed: int = None,
) -> Dict:
    """
    Chạy shopping phase cho một level
    
    Args:
        env: ShoppingEnv instance (or None if using random agent)
        agent: ShoppingAgent instance or RandomShoppingAgent
        level: Level hiện tại (1-9)
        money: Số tiền sau mining phase
        seed: Random seed for reproducibility
        
    Returns:
        Dict với 'action', 'items_bought', 'cost', 'reward', 'money_after'
    """
    # Set seed for this phase if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    
    # If using random agent, just get random action and return
    if env is None:
        # Random shopping - generate shop state and choose random feasible action
        shop_state = generate_random_shop(level)
        
        # Get valid actions based on money and shop availability
        valid_actions = get_valid_actions(shop_state, money)
        
        # Choose random action from valid actions
        if valid_actions:
            action = random.choice(valid_actions)
        else:
            action = 0  # Skip if no valid actions
        
        items_bought = decode_action(action)
        cost = calculate_cost(items_bought, shop_state.prices)
        money_after = money - cost
        
        return {
            'action': action,
            'items_bought': items_bought,
            'cost': cost,
            'reward': 0,  # No reward prediction for random
            'money_before': money,
            'money_after': money_after,
            'shop_state': shop_state,
            'level': level
        }
    
    # Normal trained agent flow
    # Reset env để level này (chỉ để get shop_state, KHÔNG chạy mining)
    obs, info = env.reset(seed=seed, options={'level': level})
    
    # Get shop state từ env.shop_state (không có trong info)
    shop_state = env.shop_state
    
    # Get action từ agent với constraints
    action = agent.get_action(obs, shop_state=shop_state, money=money)
    
    # KHÔNG gọi env.step() vì nó sẽ chạy lại mining episode
    # Chỉ decode action và tính cost trực tiếp
    items_bought = decode_action(action)
    cost = calculate_cost(items_bought, shop_state.prices)
    money_after = money - cost
    
    # Reward = 0 vì không chạy mining ở đây (mining sẽ chạy trong run_mining_phase của level tiếp theo)
    reward = 0
    
    return {
        'action': action,
        'items_bought': items_bought,
        'cost': cost,
        'reward': reward,
        'money_before': money,
        'money_after': money_after,
        'shop_state': shop_state,
        'level': level
    }


def run_full_simulation(
    mining_trainer: QtentionTrainer,
    shopping_agent: ShoppingAgent,
    shopping_env: ShoppingEnv,
    num_levels: int = 10,
    show_details: bool = False,
    base_seed: int = None,
) -> Dict:
    """
    Chạy simulation đầy đủ qua num_levels levels
    
    Args:
        mining_trainer: Mining agent trainer
        shopping_agent: Shopping agent
        shopping_env: Shopping environment
        num_levels: Number of levels to play
        show_details: Show detailed output
        base_seed: Base seed for reproducibility (each level gets base_seed + level)
    
    Returns:
        Dict với full results
    """
    results = {
        'levels': [],
        'total_gold': 0,  # Track total gold earned
        'total_money_spent': 0,
    }
    
    current_money = 0  # Bắt đầu với 0 tiền
    
    for level in range(1, num_levels + 1):
        # Generate seed for this level
        level_seed = base_seed if base_seed is not None else None
        if show_details:
            print(f"\n{'='*60}")
            print(f"LEVEL {level}/{num_levels}")
            print(f"{'='*60}")
            print(f"Money before level: ${current_money}")
        
        # ========== MINING PHASE ==========
        if show_details:
            print(f"\n[Mining Phase]")
            if level_seed is not None:
                print(f"  Seed: {level_seed}")
        
        mining_result = run_mining_phase(mining_trainer, level, seed=level_seed)
        
        if show_details:
            print(f"  Gold earned: ${mining_result['gold_earned']}")
            print(f"  Reward: {mining_result['reward']:.1f}")
            print(f"  Steps: {mining_result['steps']}")
        
        # Update money và stats (use gold_earned for actual money)
        gold_from_level = mining_result['gold_earned']
        current_money += gold_from_level
        results['total_gold'] += gold_from_level
        
        # ========== SHOPPING PHASE ==========
        if level < num_levels:  # Không có shop ở level cuối
            if show_details:
                print(f"\n[Shopping Phase]")
                print(f"  Money available: ${current_money}")
            
            shopping_result = run_shopping_phase(
                shopping_env, 
                shopping_agent, 
                level, 
                current_money,
                seed=49
            )
            
            if show_details:
                print(f"\n  {'='*50}")
                print(f"  SHOPPING DECISION (Level {level})")
                print(f"  {'='*50}")
                items_str = ', '.join([k.upper() for k, v in shopping_result['items_bought'].items() if v])
                if items_str:
                    print(f"  ✓ Items bought: {items_str}")
                else:
                    print(f"  ✗ Skipped shopping (no items bought)")
                print(f"  Action ID: {shopping_result['action']}")
                print(f"  Cost: ${shopping_result['cost']}")
                print(f"  Expected reward: {shopping_result['reward']:.1f}")
                print(f"  Money: ${shopping_result['money_before']} → ${shopping_result['money_after']}")
                print(f"  {'='*50}\n")
            
            # Update money
            current_money = shopping_result['money_after']
            results['total_money_spent'] += shopping_result['cost']
            
            # Save level results
            level_result = {
                'level': level,
                'mining': mining_result,
                'shopping': shopping_result
            }
        else:
            # Level cuối không có shopping
            level_result = {
                'level': level,
                'mining': mining_result,
                'shopping': None
            }
        
        results['levels'].append(level_result)
    
    results['final_money'] = current_money
    
    return results


def print_summary(run_results: List[Dict], num_runs: int):
    """Print summary của tất cả runs"""
    print(f"\n{'='*80}")
    print(f"SUMMARY - {num_runs} RUNS")
    print(f"{'='*80}\n")
    
    # Individual run results
    print(f"{'Run':<6} {'Total Gold':<20} {'Money Spent':<20} {'Final Money':<20}")
    print(f"{'-'*70}")
    
    for i, result in enumerate(run_results, 1):
        print(f"{i:<6} "
              f"${result['total_gold']:<19} "
              f"${result['total_money_spent']:<19} "
              f"${result['final_money']:<19}")
    
    print(f"{'-'*70}")
    
    # Statistics
    total_gold = [r['total_gold'] for r in run_results]
    money_spent = [r['total_money_spent'] for r in run_results]
    final_money = [r['final_money'] for r in run_results]
    
    print(f"\n{'METRIC':<25} {'MEAN':<15} {'STD':<15} {'MIN':<15} {'MAX':<15}")
    print(f"{'-'*80}")
    print(f"{'Total Gold Earned':<25} "
          f"${np.mean(total_gold):<14.1f} "
          f"${np.std(total_gold):<14.1f} "
          f"${np.min(total_gold):<14.1f} "
          f"${np.max(total_gold):<14.1f}")
    print(f"{'Money Spent':<25} "
          f"${np.mean(money_spent):<14.1f} "
          f"${np.std(money_spent):<14.1f} "
          f"${np.min(money_spent):<14.1f} "
          f"${np.max(money_spent):<14.1f}")
    print(f"{'Final Money':<25} "
          f"${np.mean(final_money):<14.1f} "
          f"${np.std(final_money):<14.1f} "
          f"${np.min(final_money):<14.1f} "
          f"${np.max(final_money):<14.1f}")
    
    print(f"\n{'='*80}\n")
    
    # Summary stats
    print(f"Average Gold per Level: ${np.mean(total_gold) / 10:.1f}")
    print(f"Net Profit per Run: ${np.mean(final_money):.1f}")
    print()


def save_results(run_results: List[Dict], output_path: str):
    """Save results to JSON file"""
    # Convert numpy types and custom objects to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert custom objects (like ShopState) to dict
            return convert_types(obj.__dict__)
        else:
            return obj
    
    results_json = convert_types(run_results)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run full Gold Miner simulation')
    
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of runs to execute (default: 5)')
    parser.add_argument('--mining-checkpoint', type=str, default='C:\\Users\\User\\Documents\\code\\rl-training-gold-miner\\checkpoints\\qtention\\checkpoint_cycle_2000.pth',
                        help='Path to mining agent checkpoint (default: None for random guessing)')
    parser.add_argument('--shopping-checkpoint', type=str, default='C:\\Users\\User\\Documents\\code\\rl-training-gold-miner\\checkpoints\\shopping\\final.npz',
                        help='Path to shopping agent checkpoint (default: None for random guessing)')
    parser.add_argument('--num-levels', type=int, default =10,
                        help='Number of levels per run (default: 10)')
    parser.add_argument('--show-details', action='store_true',
                        help='Show detailed output for each level')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results JSON file')
    parser.add_argument('--seed', type=int, default=49,
                        help='Random seed for reproducibility')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit (default: 60)')
    parser.add_argument('--show', type = bool, default=True,
                        help='Show game window during simulation')
    
    args = parser.parse_args()
    
    # Detect random mode from checkpoint paths
    use_random_mining = (args.mining_checkpoint is None)
    use_random_shopping = (args.shopping_checkpoint is None)
    
    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    print(f"\n{'='*80}")
    print(f"FULL SIMULATION - {args.num_runs} RUNS x {args.num_levels} LEVELS")
    print(f"{'='*80}")
    print(f"Mining agent: {'Random' if use_random_mining else args.mining_checkpoint}")
    print(f"Shopping agent: {'Random' if use_random_shopping else args.shopping_checkpoint}")
    print(f"{'='*80}\n")
    
    # Create mining environment first
    print("Creating mining environment...")
    render_mode = 'human' if args.show else None
    game_speed = (args.fps / 60.0) if args.show else 10.0  # Slower when showing, faster otherwise
    mining_env = GoldMinerEnv(
        render_mode=render_mode,
        max_steps=3600,
        levels=list(range(1, args.num_levels + 1)),  # List [1, 2, ..., num_levels] thay vì int
        use_generated_levels=False,
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
        args.mining_checkpoint, 
        device, 
        mining_env, 
        use_random=use_random_mining
    )
    
    # Load shopping agent (or random)
    shopping_agent = load_shopping_agent(
        args.shopping_checkpoint,
        use_random=use_random_shopping
    )
    
    # Create shopping environment (only if using trained shopping agent)
    if not use_random_shopping:
        print("\nCreating shopping environment...")
        shopping_env = ShoppingEnv(
            mining_env=mining_env,
            mining_trainer=mining_trainer,
            levels=list(range(1, args.num_levels + 1))
        )
        print("✓ Shopping environment created\n")
    else:
        shopping_env = None  # Not needed for random shopping
        print("\n✓ Random shopping - shopping environment not needed\n")
    
    # Run simulations
    all_results = []
    
    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'#'*80}")
        print(f"RUN {run_idx}/{args.num_runs}")
        print(f"{'#'*80}")
        
        result = run_full_simulation(
            mining_trainer=mining_trainer,
            shopping_agent=shopping_agent,
            shopping_env=shopping_env,
            num_levels=args.num_levels,
            show_details=(args.show_details or args.show),  # Always show details when --show
            base_seed=(args.seed + (run_idx - 1) * 1000) if args.seed is not None else None
        )
        
        all_results.append(result)
        
        # Print run summary
        print(f"\nRun {run_idx} Summary:")
        print(f"  Total Gold Earned: ${result['total_gold']}")
        print(f"  Money Spent: ${result['total_money_spent']}")
        print(f"  Final Money: ${result['final_money']}")
        print(f"  Net Profit: ${result['final_money']}")
    
    # Print overall summary
    print_summary(all_results, args.num_runs)
    
    # Save results if requested
    if args.save_results:
        save_results(all_results, args.save_results)
    
    print("✓ All runs completed!\n")


if __name__ == '__main__':
    main()
