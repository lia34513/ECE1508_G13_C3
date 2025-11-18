import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
import numpy as np
import os
from rule_based import fixed_speed_keep_lane
from performance_metrics import get_collision_rate, get_average_speed
from dqn_agent import load_dqn_agent_from_checkpoint


def get_action(env, method=0, agent=None, state=None):
    """Get action for the environment."""
    if method == 0:
        return fixed_speed_keep_lane(env)
    elif method == 1:
        
        agent.epsilon = 0.0
        action = agent.select_action(state)

        return action
   
    print(f"Error: Unknown method {method}.")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Testing environment')
    parser.add_argument('--env', type=str, default='highway', help='Environment name: highway or roundabout')
    parser.add_argument('--render_mode', type=str, default='rgb_array', help='Render mode: rgb_array or human')
    parser.add_argument('--duration', type=int, default=40, help='The duration of the episode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--method', type=int, default=0, help='0: fixed speed & keep lane (default)')
    parser.add_argument('--high_speed_reward_weight', type=float, default=0.4, help='Reward weight for the Speed')
    parser.add_argument('--collision_reward_weight', type=float, default=-1.0, help='Reward weight for the Collision')
    parser.add_argument('--traffic_density', type=float, default=1.0, help='The density of the traffic: 1.0 is the default, 1.25 is the high density')
    return parser.parse_args()


def main():
    """Main function to run environment testing."""
    args = parse_args()
    
    # Print settings at the beginning
    print(f"=== Testing Settings ===")
    print(f"Environment: {args.env}")
    print(f"Duration: {args.duration}s")
    print(f"Epochs: {args.epochs}")
    print(f"Method: {args.method}")
    print(f"Traffic Density: {args.traffic_density}")
    print()
    
    # Set the configuration Reward function of the environment. https://github.com/Farama-Foundation/HighwayEnv/blob/b9180dfaef13c3c87eeb43f56f37b0e42d9d0476/highway_env/envs/highway_env.py
    config = {
        "collision_reward": args.collision_reward_weight,          # Penalty for collisions
        "high_speed_reward": args.high_speed_reward_weight,        # Coefficient for velocity
        "right_lane_reward": 0.0,        # Coefficient for lane preference
        "reward_speed_range": [20, 30],  # v_min and v_max for normalization
        "normalize_reward": True,         # Optional normalization to [0, 1]
        "vehicles_density": args.traffic_density, # The density of the traffic
        "duration": args.duration, # The duration of the episode
    }

    # Create environment based on user selection 
    if args.env == "highway":
        env = gymnasium.make('highway-v0', render_mode=args.render_mode, config=config)
    elif args.env == "roundabout":
        env = gymnasium.make('roundabout-v0', render_mode=args.render_mode, config=config)
    else:
        print(f"Error: Unknown environment '{args.env}'. Available options: highway, roundabout")
        return
    
    # Initialize DQN agent if method 1 is selected
    agent = None
    if args.method == 1:
        # Use static checkpoint path
        checkpoint_path = os.path.join("model", "DQN", "Checkpoint", "dqn_highway.pth")
        
        # Get state dimensions from environment
        obs, info = env.reset()
        state = np.array(obs.get("observation", obs) if isinstance(obs, dict) else obs, dtype=np.float32).flatten()
        state_dim = state.shape[0]
        action_dim = env.action_space.n
        
        # Load DQN agent from checkpoint
        agent = load_dqn_agent_from_checkpoint(state_dim, action_dim, checkpoint_path)
  
    
    # Track collisions and speeds
    total_collisions = 0
    total_epochs = 0
    collision_rate = 0
    episode_speeds = []
    episode_rewards = []

    for epoch_num in range(args.epochs):
        obs, info = env.reset()
        state = np.array(obs.get("observation", obs) if isinstance(obs, dict) else obs, dtype=np.float32).flatten() if args.method == 1 else None
        
        epoch_reward = 0.0
        epoch_steps = 0
        step_speeds = []

        # Run the trajectory until the episode is done or terminated (i.e. crashed or reached the duration)
        while True:
            action = get_action(env, args.method, agent=agent, state=state)
            if action is None:
                print(f"Error: Failed to get action for method {args.method}. Exiting.")
                return
            obs, reward, done, truncated, info = env.step(action)
            
            # Update state for DQN agent
            if args.method == 1:
                state = np.array(obs.get("observation", obs) if isinstance(obs, dict) else obs, dtype=np.float32).flatten()
            
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
