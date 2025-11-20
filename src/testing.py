import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
import numpy as np
import os
from rule_based import fixed_speed_keep_lane, OPDAgent
from performance_metrics import get_collision_rate, get_average_speed
from env_config import get_highway_config
from stable_baselines3 import DQN


def get_action(env, method=0, model=None, agent=None, obs=None):
    """Get action for the environment.

    Args:
        env: The gymnasium environment
        method: The policy method to use
            0: fixed_speed_keep_lane (baseline)
            1: stable-baselines3 DQN
            2: OPD (Optimistic Planning)
        model: DQN model instance (required for method 1)
        agent: OPD agent instance (required for method 2)
        obs: Current observation (required for methods 1 and 2)
    """
    if method == 0:
        return fixed_speed_keep_lane(env)
    elif method == 1:
        # DQN policy
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    elif method == 2:
        # OPD policy
        if agent is None or obs is None:
            print("Error: OPD method requires agent and observation.")
            return None
        return agent.act(obs)

    print(f"Error: Unknown method {method}.")
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Testing environment")
    parser.add_argument(
        "--env",
        type=str,
        default="highway",
        help="Environment name: highway or roundabout",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        help="Render mode: rgb_array or human",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--method",
        type=int,
        default=0,
        help="0: fixed speed & keep lane, 1: stable-baselines3 DQN, 2: OPD (Optimistic Planning)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="The duration of the episode (if not set, uses env_config defaults)",
    )
    parser.add_argument(
        "--high_speed_reward_weight",
        type=float,
        default=None,
        help="Reward weight for the Speed (if not set, uses env_config defaults)",
    )
    parser.add_argument(
        "--collision_reward_weight",
        type=float,
        default=None,
        help="Reward weight for the Collision (if not set, uses env_config defaults)",
    )
    parser.add_argument(
        "--traffic_density",
        type=float,
        default=None,
        help="The density of the traffic (if not set, uses env_config defaults)",
    )
    # OPD specific parameters
    parser.add_argument(
        "--opd_budget",
        type=int,
        default=50,
        help="OPD planning budget (number of expansions)",
    )
    parser.add_argument(
        "--opd_gamma", type=float, default=0.7, help="OPD discount factor"
    )
    return parser.parse_args()


def main():
    """Main function to run environment testing."""
    args = parse_args()

    # Map method numbers to names
    method_names = {
        0: "Fixed Speed & Keep Lane",
        1: "DQN (Deep Q-Network)",
        2: "OPD (Optimistic Planning)",
    }
    method_name = method_names.get(args.method, f"Unknown ({args.method})")

    # Set the configuration of the environment (using defaults from env_config)
    config = get_highway_config()

    # Override with command-line arguments if provided
    if args.high_speed_reward_weight is not None:
        config["high_speed_reward"] = args.high_speed_reward_weight
    if args.collision_reward_weight is not None:
        config["collision_reward"] = args.collision_reward_weight
    if args.traffic_density is not None:
        config["vehicles_density"] = args.traffic_density
    if args.duration is not None:
        config["duration"] = args.duration

    # Print settings at the beginning
    print(f"=== Testing Settings ===")
    print(f"Environment: {args.env}")
    print(f"Duration: {config['duration']}s")
    print(f"Epochs: {args.epochs}")
    print(f"Method: {method_name}")
    print(f"Traffic Density: {config['vehicles_density']}")
    print(f"High Speed Reward Weight: {config['high_speed_reward']}")
    print(f"Collision Reward Weight: {config['collision_reward']}")
    if args.method == 2:
        print(f"OPD Budget: {args.opd_budget}")
        print(f"OPD Gamma: {args.opd_gamma}")
    print()

    # Create environment based on user selection
    if args.env == "highway":
        env = gymnasium.make("highway-v0", render_mode=args.render_mode, config=config)
    elif args.env == "roundabout":
        env = gymnasium.make(
            "roundabout-v0", render_mode=args.render_mode, config=config
        )
    else:
        print(
            f"Error: Unknown environment '{args.env}'. Available options: highway, roundabout"
        )
        return

    # Initialize model/agent based on method
    DQN_model = None
    OPD_agent = None

    if args.method == 1:
        # Stable-baselines3 DQN model
        checkpoint_path = os.path.join(
            "model",
            "DQN",
            "checkpoints",
            f"dqn_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}",
        )

        # Load stable-baselines3 model
        DQN_model = DQN.load(checkpoint_path, env=env)
        print(f"DQN model loaded from: {checkpoint_path}\n")

    elif args.method == 2:
        # OPD agent
        OPD_agent = OPDAgent(env, budget=args.opd_budget, gamma=args.opd_gamma)
        print(
            f"OPD Agent initialized with budget={args.opd_budget}, gamma={args.opd_gamma}\n"
        )

    # Track collisions and speeds
    total_collisions = 0
    total_epochs = 0
    collision_rate = 0
    episode_speeds = []
    episode_total_rewards = []  # Total reward per episode
    episode_avg_rewards = []    # Average reward per step per episode

    for epoch_num in range(args.epochs):
        obs, info = env.reset()

        epoch_reward = 0.0
        epoch_steps = 0
        step_speeds = []

        # Run the trajectory until the episode is done or terminated (i.e. crashed or reached the duration)
        while True:
            action = get_action(
                env, args.method, model=DQN_model, agent=OPD_agent, obs=obs
            )
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
            if args.render_mode == "human":
                env.render()

            if done or truncated:
                break

        # Save episode speed data
        episode_speeds.append(step_speeds)

        # Check for collision at the end of the epoch
        total_epochs += 1
        crashed = info.get("crashed", False)
        if crashed:
            total_collisions += 1

        avg_reward = epoch_reward / epoch_steps if epoch_steps > 0 else 0.0
        episode_total_rewards.append(epoch_reward)
        episode_avg_rewards.append(avg_reward)

        # Display crashed state and performance for each epoch
        crashed_status = "CRASHED" if crashed else "OK"
        print(
            f"Epoch {epoch_num + 1}/{args.epochs}: {crashed_status}, Total reward: {epoch_reward:.2f}, Avg reward: {avg_reward:.4f}"
        )

    # Calculate and display metrics
    collision_rate = get_collision_rate(total_collisions, total_epochs)
    average_speed = get_average_speed(episode_speeds)
    avg_total_reward = np.mean(episode_total_rewards) if episode_total_rewards else 0.0
    avg_per_step_reward = np.mean(episode_avg_rewards) if episode_avg_rewards else 0.0

    print(f"\n=== Testing Results ===")
    print(f"Method: {method_name}")
    print(f"Total epochs: {total_epochs}")
    print(f"Collisions: {total_collisions}")

    if collision_rate is not None:
        print(f"Collision Rate: {collision_rate:.2f}%")
    else:
        print(
            "Error: Collision rate is None. Please check the total_collisions and total_epochs."
        )

    if average_speed is not None:
        print(f"Average Speed: {average_speed:.2f} m/s")
    else:
        print("Error: Average speed is None. Please check the episode_speeds.")

    print(f"Average Total Reward (per episode): {avg_total_reward:.4f}")
    print(f"Average Reward (per step): {avg_per_step_reward:.4f}")

    # Print summary for easy copy-paste to table
    print(f"\n=== Summary for Table ===")
    print(
        f"Collision Rate: {collision_rate:.2f}%"
        if collision_rate is not None
        else "Collision Rate: N/A"
    )
    print(
        f"Average Speed: {average_speed:.2f} m/s"
        if average_speed is not None
        else "Average Speed: N/A"
    )
    print(f"Average Total Reward: {avg_total_reward:.4f}")
    print(f"Average Per-Step Reward: {avg_per_step_reward:.4f}")


if __name__ == "__main__":
    main()
