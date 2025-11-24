import gymnasium
import highway_env
from matplotlib import pyplot as plt
import argparse
import numpy as np
import os
from rule_based import OPDAgent
from performance_metrics import get_collision_rate, get_collision_rate_per_action, get_average_speed
from env_config import get_highway_config, parse_args
from stable_baselines3 import DQN, PPO


def get_action(env, method=0, agent=None, obs=None):
    """Get action for the environment.

    Args:
        env: The gymnasium environment
        method: The policy method to use
            0: OPD (Optimistic Planning)
            1: stable-baselines3 DQN
            2: stable-baselines3 PPO
        agent: Agent instance (required for all methods)
        obs: Current observation (required for all methods)
    """
    if agent is None or obs is None:
        print("Error: The method requires agent and observation.")
        return None

    if method == 0:
        # OPD policy
        return agent.act(obs)
    elif method == 1:
        # DQN policy
        action, _ = agent.predict(obs, deterministic=True)
        return int(action)
    elif method == 2:
        # PPO policy
        action, _ = agent.predict(obs, deterministic=True)
        return int(action)

    print(f"Error: Unknown method {method}.")
    return None

def main():
    """Main function to run environment testing."""
    args = parse_args()

    # Map method numbers to names
    method_names = {
        0: "OPD (Optimistic Planning)",
        1: "DQN (Deep Q-Network)",
        2: "PPO (Proximal Policy Optimization)",
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

    # Fixed random seed for reproducibility and fair comparison across agents
    # All agents will see the same sequence of initial states for each epoch
    fixed_seed = 42
    
    # Print settings at the beginning
    print(f"=== Testing Settings ===")
    print(f"Environment: {args.env}")
    print(f"Duration: {config['duration']}s")
    print(f"Epochs: {args.epochs}")
    print(f"Method: {method_name}")
    print(f"Random Seed: {fixed_seed} (deterministic per epoch for fair comparison across agents)")
    print(f"Traffic Density: {config['vehicles_density']}")
    print(f"High Speed Reward Weight: {config['high_speed_reward']}")
    print(f"Collision Reward Weight: {config['collision_reward']}")
    if args.method == 0:
        print(f"OPD Budget: {args.opd_budget}")
        print(f"OPD Gamma: {args.opd_gamma}")
    print()

    # Create environment based on user selection
    # Set seed for environment initialization
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
    
    # Set the random seed for numpy and the environment for reproducibility
    np.random.seed(fixed_seed)
    env.reset(seed=fixed_seed)

    # Initialize agent based on method
    agent = None

    if args.method == 0:
        # OPD agent
        agent = OPDAgent(env, budget=args.opd_budget, gamma=args.opd_gamma)
        print(
            f"OPD Agent initialized with budget={args.opd_budget}, gamma={args.opd_gamma}\n"
        )

    elif args.method == 1:
        checkpoint_path = os.path.join(
            "model", "DQN", "checkpoints",
            f"dqn_highway_vehicles_density_{config['vehicles_density']}"
            f"_high_speed_reward_{config['high_speed_reward']}"
            f"_collision_reward_{config['collision_reward']}"
        )
        print(f"[TEST] Loading DQN from: {checkpoint_path}")

        # Load stable-baselines3 DQN agent
        agent = DQN.load(checkpoint_path, env=env)

    elif args.method == 2:
        checkpoint_path = os.path.join(
            "model", "PPO", "checkpoints",
            f"ppo_highway_vehicles_density_{config['vehicles_density']}"
            f"_high_speed_reward_{config['high_speed_reward']}"
            f"_collision_reward_{config['collision_reward']}.zip"
        )

        print(f"[TEST] Loading PPO from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] PPO checkpoint not found: {checkpoint_path}")
            return

        # Load stable-baselines3 PPO agent
        agent = PPO.load(checkpoint_path, env=env)

    total_collisions = 0
    total_epochs = 0
    total_actions = 0
    total_reward = 0.0
    collision_rate_per_episode = 0
    collision_rate_per_action = 0
    episode_speeds = []
    episode_total_rewards = []  # Total reward per episode

    for epoch_num in range(args.epochs):
        # Use deterministic seed for each epoch based on epoch number
        # This ensures all agents see the same sequence of environments
        epoch_seed = fixed_seed + epoch_num
        obs, info = env.reset(seed=epoch_seed)

        epoch_reward = 0.0
        epoch_steps = 0
        step_speeds = []

        # Run the trajectory until the episode is done or terminated (i.e. crashed or reached the duration)
        while True:
            action = get_action(env, args.method, agent=agent, obs=obs)
            if action is None:
                print(f"Error: Failed to get action for method {args.method}. Exiting.")
                return
            obs, reward, done, truncated, info = env.step(action)

            epoch_reward += reward
            total_reward += reward
            epoch_steps += 1
            total_actions += 1

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

        episode_total_rewards.append(epoch_reward)

        # Display crashed state and performance for each epoch
        avg_reward_per_action = epoch_reward / epoch_steps if epoch_steps > 0 else 0.0
        crashed_status = "CRASHED" if crashed else "OK"
        print(
            f"Epoch {epoch_num + 1}/{args.epochs}: {crashed_status}, Total reward: {epoch_reward:.2f}, Avg reward per action: {avg_reward_per_action:.4f}"
        )

    # Calculate and display metrics
    collision_rate_per_episode = get_collision_rate(total_collisions, total_epochs)
    collision_rate_per_action = get_collision_rate_per_action(total_collisions, total_actions)
    average_speed = get_average_speed(episode_speeds)
    avg_reward_per_episode = np.mean(episode_total_rewards) if episode_total_rewards else 0.0
    avg_reward_per_action = total_reward / total_actions if total_actions > 0 else 0.0


    # Print summary for easy copy-paste to table
    print(f"\n=== Testing Results ===")
    print(
        f"Collision Rate (per episode): {collision_rate_per_episode:.2f}%"
        if collision_rate_per_episode is not None
        else "Collision Rate (per episode): N/A"
    )
    print(
        f"Collision Rate (per action): {collision_rate_per_action:.4f}%"
        if collision_rate_per_action is not None
        else "Collision Rate (per action): N/A"
    )
    print(f"Average Reward (per episode): {avg_reward_per_episode:.4f}")
    print(f"Average Reward (per action): {avg_reward_per_action:.4f}")
    print(
        f"Average Speed: {average_speed:.2f} m/s"
        if average_speed is not None
        else "Average Speed: N/A"
    )


if __name__ == "__main__":
    main()
