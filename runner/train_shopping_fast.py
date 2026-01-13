import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import math
import pickle
from tqdm import tqdm
import random
from agent.ShoppingAgent import ShoppingAgent
from model.ShoppingEnv import decode_action
import optuna
import numpy as np
import argparse
import pygame
from typing import Dict, List, Tuple
import time
from agent.Qtention.Qtention import Qtention
from agent.ShoppingAgent import ShoppingAgent
from trainer.QtentionTrainer import QtentionTrainer, angle_bins
from model.GoldMiner import GoldMinerEnv
from model.ShoppingEnv import ShoppingEnv, generate_random_shop, get_valid_actions, decode_action, calculate_cost
from runner.full_run import run_full_simulation, load_mining_agent

buffer = None

def train_shopping_heuristic(R_dyn):
    agent = ShoppingAgent()
    agent.set_train_mode(epsilon=0.1)
    num_episodes = 10000
    for level in range(1, 10):
        for episode in range(num_episodes):
                action = agent.get_action({'level': level})
                items_bought = decode_action(action)
                play = random.choice(buffer[level])
                reward = play['other'] + play['rock'] * (3 if items_bought['rock'] else 1) + int(play['diamond'] * (1.5 if items_bought['gem'] else 1)) + items_bought['dynamite'] * R_dyn[level]
                agent.update(level, action, reward)
    return agent

parser = argparse.ArgumentParser(description='Run full Gold Miner simulation')
parser.add_argument('--num-runs', type=int, default=5,
                    help='Number of runs to execute per game (default: 5)')
parser.add_argument('--checkpoint', type=str, default='/mnt/disk1/aiotlab/namth/rl-training-gold-miner/checkpoints/qtention/checkpoint_cycle_2000.pth',
                    help='Path to mining agent checkpoint (default: None for random guessing)')
parser.add_argument('--fps', type=int, default=180,
                    help='FPS limit (default: 180)')
parser.add_argument('--show', action='store_true',
                    help='Show game window during simulation')
parser.add_argument('--net', type=str, default='attention',
                        help="Type of shopping agent network: 'attention', 'cnn', or 'cnn_rnn' (default: 'attention')")
parser.add_argument('--buffer-path', type=str, default='/mnt/disk1/aiotlab/namth/rl-training-gold-miner/buffers/shopping_buffer.pkl',
                    help='Path to shopping experience buffer (default: shopping_buffer.pkl)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

use_random_mining = (args.checkpoint is None)
    
mining_trainer = load_mining_agent(
    args.net,
    args.checkpoint, 
    device, 
    mining_env, 
    use_random=use_random_mining
)

shopping_env = ShoppingEnv(
    mining_env=mining_env,
    mining_trainer=mining_trainer,
    levels=list(range(1, 11))
)

def full_evaluate(agent):
    total_gold = []
    total_money_spent = []
    final_money = []
    
    for run in range(args.num_runs):
        result = run_full_simulation(
            mining_trainer,
            agent,
            shopping_env,
            show_details=False,
            show_window = True if args.show == True else False,
        )
        
        final_money.append(result['final_money'])       
        total_gold.append(result['total_gold'])
        total_money_spent.append(result['total_money_spent'])
    return total_gold, total_money_spent, final_money

def evaluate_blackbox(x: np.ndarray) -> float:
    assert x.shape == (9,)
    x = np.concatenate(([0], x))  # Add dummy for level 0
    agent = train_shopping_heuristic(x)
    agent.set_eval_mode()
    total_gold, total_money_spent, final_money = full_evaluate(agent)
    return np.mean(final_money) / 1000.0

buffer = pickle.load(open(args.buffer_path, "rb"))

D = 9
LOW, HIGH = 0, 2000

def objective(trial: optuna.Trial) -> float:
    # Khai báo đúng kiểu integer, KHÔNG cần scale/round
    x = np.array([trial.suggest_int(f"x{i}", LOW, HIGH) for i in range(D)], dtype=np.int64)

    # Nếu muốn chắc chắn truyền float vào blackbox:
    # y = evaluate_blackbox(x.astype(np.float64))
    y = evaluate_blackbox(x)

    return float(y)  # minimize

sampler = optuna.samplers.GPSampler(
    seed=42,
    n_startup_trials=20,          # thường 10–30 cho 10D
    deterministic_objective=False, # True nếu hàm gần như deterministic; noisy thì để False
    # independent_sampler=...      # (tuỳ chọn) sampler dùng cho startup/conditional params
)

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)

print("best value:", study.best_value)
print("best params:", study.best_params)

print("Done!")