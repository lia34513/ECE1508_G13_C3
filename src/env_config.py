import argparse

def get_highway_config():
    """
    Create configuration dictionary for highway-env environments.
    
    Reference: https://github.com/Farama-Foundation/HighwayEnv/blob/b9180dfaef13c3c87eeb43f56f37b0e42d9d0476/highway_env/envs/highway_env.py
    """
    config = {
        "collision_reward": -1,          # Penalty for collisions
        "high_speed_reward": 0.4,        # Coefficient for velocity (choose from 0.25, 0.5, and 0.75)
        "right_lane_reward": 0,          # Coefficient for lane preference
        "reward_speed_range": [20, 30],  # v_min and v_max for normalization
        "normalize_reward": True,        # Optional normalization to [0, 1]
        "vehicles_density": 1.25,           # The density of the traffic
        "duration": 40,                  # The duration of the episode
    }
    return config

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
        help="0: OPD (Optimistic Planning), 1: stable-baselines3 DQN, 2: stable-baselines3 PPO",
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