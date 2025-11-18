def get_highway_config():
    """
    Create configuration dictionary for highway-env environments.
    
    Reference: https://github.com/Farama-Foundation/HighwayEnv/blob/b9180dfaef13c3c87eeb43f56f37b0e42d9d0476/highway_env/envs/highway_env.py
    """
    config = {
        "collision_reward": -1,          # Penalty for collisions
        "high_speed_reward": 0.4,        # Coefficient for velocity
        "right_lane_reward": 0,           # Coefficient for lane preference
        "reward_speed_range": [20, 30],            # v_min and v_max for normalization
        "normalize_reward": True,                 # Optional normalization to [0, 1]
        "vehicles_density": 1,                 # The density of the traffic
        "duration": 40,                        # The duration of the episode
    }
    return config

