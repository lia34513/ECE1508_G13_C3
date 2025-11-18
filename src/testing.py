import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
import numpy as np
import os
from rule_based import fixed_speed_keep_lane
from performance_metrics import get_collision_rate, get_average_speed
from env_config import get_highway_config
from stable_baselines3 import DQN
import torch

def get_action(env, method=0, model=None, obs=None):
    """Get action for the environment."""
    if method == 0:
        return fixed_speed_keep_lane(env)
    elif method == 1:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
   
    print(f"Error: Unknown method {method}.")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Testing environment')
    parser.add_argument('--env', type=str, default='highway', help='Environment name: highway or roundabout')
    parser.add_argument('--render_mode', type=str, default='rgb_array', help='Render mode: rgb_array or human')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--method', type=int, default=0, help='0: fixed speed & keep lane (default), 1: stable-baselines3 DQN')
    return parser.parse_args()


def main():
    """Main function to run environment testing."""
    args = parse_args()
    
    # Set the configuration of the environment (using defaults from env_config)
    config = get_highway_config()
    
    # Print settings at the beginning
    print(f"=== Testing Settings ===")
    print(f"Environment: {args.env}")
    print(f"Duration: {config['duration']}s")
    print(f"Epochs: {args.epochs}")
    print(f"Method: {args.method}")
    print(f"Traffic Density: {config['vehicles_density']}")
    print(f"Collision Reward Weight: {config['collision_reward']}")
    print(f"High Speed Reward Weight: {config['high_speed_reward']}")
    print()

    # Create environment based on user selection 
    if args.env == "highway":
        env = gymnasium.make('highway-v0', render_mode=args.render_mode, config=config)
    elif args.env == "roundabout":
        env = gymnasium.make('roundabout-v0', render_mode=args.render_mode, config=config)
    else:
        print(f"Error: Unknown environment '{args.env}'. Available options: highway, roundabout")
        return
    
    # Initialize model based on method
    DQN_model = None
    
    if args.method == 1:
        # Stable-baselines3 DQN model
        checkpoint_path = os.path.join(
            "model", "DQN", "checkpoints",
            f"dqn_highway_vehicles_density_{config['vehicles_density']}_"
            f"high_speed_reward_{config['high_speed_reward']}_"
            f"collision_reward_{config['collision_reward']}.zip"
        )
        
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print("Testing using device:", device)

        # Load stable-baselines3 model
        DQN_model = DQN.load(checkpoint_path, env=env, device=device)
  
    
    # Track collisions and speeds
    total_collisions = 0
    total_epochs = 0
    collision_rate = 0
    episode_speeds = []
    episode_rewards = []

    for epoch_num in range(args.epochs):
        obs, info = env.reset()
        
        epoch_reward = 0.0
        epoch_steps = 0
        step_speeds = []

        # Run the trajectory until the episode is done or terminated (i.e. crashed or reached the duration)
        while True:
            action = get_action(env, args.method, model=DQN_model, obs=obs)
            if action is None:
                print(f"Error: Failed to get action for method {args.method}. Exiting.")
                return
            obs, reward, done, truncated, info = env.step(action)
            
            epoch_reward += reward
            epoch_steps += 1
          
            # Record the ego vehicle speed if available in info.
            ego_speed = info.get("speed")
            if ego_speed is not None:
                step_speeds.append(ego_speed)

            # Only render if using human mode
            if args.render_mode == 'human':
                env.render()

            if done or truncated:
                break

        # Save episode speed data
        episode_speeds.append(step_speeds)

        # Check for collision at the end of the epoch
        total_epochs += 1
        crashed = info.get('crashed', False)
        if crashed:
            total_collisions += 1

        avg_reward = epoch_reward / epoch_steps if epoch_steps > 0 else 0.0
        episode_rewards.append(avg_reward)

        # Display crashed state and performance for each epoch
        crashed_status = "CRASHED" if crashed else "OK"
        print(f"Epoch {epoch_num + 1}/{args.epochs}: {crashed_status}, Average reward: {avg_reward:.4f}")

    # Calculate and display metrics
    collision_rate = get_collision_rate(total_collisions, total_epochs)
    average_speed = get_average_speed(episode_speeds)
    overall_avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

    print(f"\n=== Testing Results ===")
    print(f"Total epochs: {total_epochs}")
    print(f"Collisions: {total_collisions}")

    print(f"Collision rate: {collision_rate:.2f}%")
    print(f"Average speed: {average_speed:.2f} (m/s)")
    print(f"Average reward: {overall_avg_reward:.2f}")


if __name__ == "__main__":
    main()
