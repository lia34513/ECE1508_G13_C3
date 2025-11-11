import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
from rule_based import fixed_speed_keep_lane
from performance_metrics import get_collision_rate, get_average_speed


def get_action(env, method=0):
    """Get action for the environment."""
    if method == 0:
        return fixed_speed_keep_lane(env)
   
    print(f"Error: Unknown method {method}.")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Testing environment')
    parser.add_argument('--env', type=str, default='highway', help='Environment name: highway or roundabout')
    parser.add_argument('--render_mode', type=str, default='rgb_array', help='Render mode: rgb_array or human')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--method', type=int, default=0, help='0: fixed speed & keep lane (default)')
    parser.add_argument('--high_speed_reward_weight', type=float, default=1.0, help='Reward weight for the Speed')
    parser.add_argument('--collision_reward_weight', type=float, default=-1.0, help='Reward weight for the Collision')
    parser.add_argument('--traffic_density', type=float, default=1.0, help='The density of the traffic: 1.0 is the default, 1.25 is the high density')
    return parser.parse_args()


def main():
    """Main function to run environment testing."""
    args = parse_args()
    
    # Print settings at the beginning
    print(f"=== Testing Settings ===")
    print(f"Environment: {args.env}")
    print(f"Steps per epoch: {args.steps}")
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
        "vehicles_density": args.traffic_density,
    }

    # Create environment based on user selection 
    if args.env == "highway":
        env = gymnasium.make('highway-v0', render_mode=args.render_mode, config=config)
    elif args.env == "roundabout":
        env = gymnasium.make('roundabout-v0', render_mode=args.render_mode, config=config)
    else:
        print(f"Error: Unknown environment '{args.env}'. Available options: highway, roundabout")
        return
    
    # Track collisions and speeds
    total_collisions = 0
    total_epochs = 0
    collision_rate = 0
    episode_speeds = []

    for epoch_num in range(args.epochs):
        obs, info = env.reset()
        
        epoch_reward = 0.0
        epoch_steps = 0
        step_speeds = []

        for _ in range(args.steps):
            action = get_action(env, args.method)
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

        # Display crashed state and performance for each epoch
        crashed_status = "CRASHED" if crashed else "OK"
        print(f"Epoch {epoch_num + 1}/{args.epochs}: {crashed_status}, Average reward: {avg_reward:.2f}")

    # Calculate and display metrics
    collision_rate = get_collision_rate(total_collisions, total_epochs)
    average_speed = get_average_speed(episode_speeds)

    print(f"\n=== Testing Results ===")
    print(f"Total epochs: {total_epochs}")
    print(f"Collisions: {total_collisions}")

    if collision_rate is not None:
        print(f"Collision rate: {collision_rate:.2f}%")
    else:
        print("Error: Collision rate is None. Please check the total_collisions and total_epochs.")

    if average_speed is not None:
        print(f"Average speed: {average_speed:.2f} (m/s)")
    else:
        print("Error: Average speed is None. Please check the episode_speeds.")


if __name__ == "__main__":
    main()
