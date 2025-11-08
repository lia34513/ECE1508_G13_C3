"""
Rule-based baseline policies for autonomous driving.
"""


def fixed_speed_keep_lane(env):
    """Fixed speed keep lane action - maintains current lane and speed (IDLE)."""
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    return action
