"""
Evaluation script để xem agent chơi Gold Miner
Load checkpoint và render game với 60 FPS
"""

import torch
import pygame
import argparse
import time
from model.GoldMiner import GoldMinerEnv
from agent.Qtention import Qtention


def evaluate_agent(checkpoint_path: str, num_episodes: int = 5, fps: int = 60):
    """
    Load agent từ checkpoint và chơi game với visualization
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Số episodes để chơi
        fps: Frames per second (giới hạn FPS)
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment với render mode
    print("\nCreating environment...")
    env = GoldMinerEnv(
        render_mode='human',  # Hiển thị game
        max_steps=3600,       # 60 giây * 60 FPS
        level=1,
        use_generated_levels=True,
        c_dyna=10,
        c_step=0.0,
        c_pull=0.3,
        reward_scale=1000.0,
        game_speed=1
    )
    print("✓ Environment created")
    
    # Create agent
    print("\nCreating agent...")
    agent = Qtention(
        d_model=20,
        n_actions=2,
        nhead=4,
        n_layers=2,
        d_ff=24,
        dropout=0.01,
        max_items=30
    ).to(device)
    
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"✓ Agent created")
    print(f"  Total parameters: {total_params:,}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()  # Set to evaluation mode
    print("✓ Checkpoint loaded")
    
    if 'total_steps' in checkpoint:
        print(f"  Training steps: {checkpoint['total_steps']}")
    if 'epsilon' in checkpoint:
        print(f"  Final epsilon: {checkpoint['epsilon']:.4f}")
    
    # Clock for FPS control
    clock = pygame.time.Clock()
    
    print("\n" + "="*60)
    print("Starting evaluation...")
    print(f"Playing {num_episodes} episodes with {fps} FPS limit")
    print("Press ESC or close window to stop")
    print("="*60 + "\n")
    
    episode_rewards = []
    episode_scores = []
    
    try:
        for episode in range(1, num_episodes + 1):
            print(f"\n{'='*60}")
            print(f"Episode {episode}/{num_episodes}")
            print('='*60)
            
            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nWindow closed by user")
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\nESC pressed by user")
                            raise KeyboardInterrupt
                
                # Select action (greedy - no exploration)
                # with torch.no_grad():
                #     q_values = agent(state)
                #     if q_values[0] < q_values[1]:
                #         print(q_values[0], q_values[1])
                #     action = q_values.argmax().item()
                action = select_action_with_forced_rules(agent, state, device)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Limit FPS
                clock.tick(fps)
            
            # Episode completed
            # final_score = info.get('final_score', 0) if 'final_score' in info else env.game_scene.score
            episode_rewards.append(episode_reward)
            
            print(f"\n{'='*60}")
            print(f"Episode {episode} completed!")
            # print(f"  Score: {final_score}")
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Steps: {episode_steps}")
            print('='*60)
            
            # Wait a bit before next episode
            if episode < num_episodes:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)
    
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
        print(f"Average score: {sum(episode_scores)/len(episode_scores):.2f}")
        print(f"Best score: {max(episode_scores)}")
        print(f"Worst score: {min(episode_scores)}")
        print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
        print("="*60)
    
    print("\n✓ Evaluation completed")


def select_action_with_forced_rules(agent, state, device):
    """
    Select action với forced action rules (giống như training)
    
    Returns:
        action: Selected action (0 or 1)
    """
    # Check rope state
    rope_state = state['rope_state']
    is_retracting_with_item = (rope_state['state'] == 'retracting')
    is_expanding = (rope_state['state'] == 'expanding')
    is_swinging = (rope_state['state'] == 'swinging')
    has_dynamite = state['global_state']['dynamite_count'] > 0
    has_item = rope_state['has_item']
    rope_timer = rope_state.get('timer', -1)
    
    # AUTO ACTION 1: Nếu đang kéo mà không có dynamite → chỉ có action 0
    if is_retracting_with_item and (not has_dynamite or not has_item):
        return 0
    
    # AUTO ACTION 2: Nếu móc đang được thả xuống → chỉ có action 0
    if is_expanding:
        return 0
    
    # AUTO ACTION 3: Nếu móc đang swinging nhưng timer > 0 (cooldown) → chỉ có action 0
    if is_swinging and rope_timer > 0:
        return 0
    
    # Use model for decision
    with torch.no_grad():
        q_values = agent(state)
        if q_values[0] < q_values[1]:
            print(q_values[0], q_values[1])
        return q_values.argmax().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent for Gold Miner')
    parser.add_argument('--checkpoint', type=str, default="C:\\Users\\User\\Downloads\\vaegan\\checkpoint.pt",
                        help='Path to checkpoint file (default: "C:\\Users\\User\\Downloads\\vaegan\\checkpoint.pt")')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit (default: 60)')
    
    args = parser.parse_args()
    
    evaluate_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        fps=args.fps
    )
