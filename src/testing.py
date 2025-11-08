import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
from rule_based import fixed_speed_keep_lane


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
    print()
    
    # Create environment based on user selection
    if args.env == "highway":
        env = gymnasium.make('highway-v0', render_mode=args.render_mode)
    elif args.env == "roundabout":
        env = gymnasium.make('roundabout-v0', render_mode=args.render_mode)
    else:
        print(f"Error: Unknown environment '{args.env}'. Available options: highway, roundabout")
        return
    
    # Track collisions
    total_collisions = 0
    total_epochs = 0
    
    for epoch_num in range(args.epochs):
        obs, info = env.reset()
        
        for _ in range(args.steps):
            action = get_action(env, args.method)
            if action is None:
                print(f"Error: Failed to get action for method {args.method}. Exiting.")
                return
            obs, reward, done, truncated, info = env.step(action)
            
            # Only render if using human mode
            if args.render_mode == 'human':
                env.render()

            if done or truncated:
                break

        # Check for collision at the end of the epoch
        total_epochs += 1
        crashed = info.get('crashed', False)
        if crashed:
            total_collisions += 1

        # Display crashed state for each epoch
        crashed_status = "CRASHED" if crashed else "OK"
        print(f"Epoch {epoch_num + 1}/{args.epochs}: {crashed_status}")

    # Calculate and display collision rate
    collision_rate = (total_collisions / total_epochs) * 100 if total_epochs > 0 else 0

    print(f"\n=== Testing Results ===")
    print(f"Total epochs: {total_epochs}")
    print(f"Collisions: {total_collisions}")
    print(f"Collision rate: {collision_rate:.2f}%")


if __name__ == "__main__":
    main()