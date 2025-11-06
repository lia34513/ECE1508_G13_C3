"""
Rule-based baseline policies for autonomous driving.
"""

import random


def fixed_speed_keep_lane(env):
    """Fixed speed keep lane action - maintains current lane and speed (IDLE)."""
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    return action


def fixed_speed_random_lane(env):
    """Fixed speed random lane action - randomly changes lanes or stays in current lane."""
    # Randomly choose between IDLE, LANE_LEFT, and LANE_RIGHT
    actions = ["IDLE", "LANE_LEFT", "LANE_RIGHT"]
    selected_action = random.choice(actions)
    action = env.unwrapped.action_type.actions_indexes[selected_action]
    return action


def random_speed_keep_lane(env):
    """Random speed keep lane action - randomly changes speed while maintaining current lane."""
    # Randomly choose between IDLE, FASTER, and SLOWER
    actions = ["IDLE", "FASTER", "SLOWER"]
    selected_action = random.choice(actions)
    action = env.unwrapped.action_type.actions_indexes[selected_action]
    return action


def random_speed_random_lane(env):
    """Random speed and random lane action - randomly changes both speed and lane."""
    # Randomly choose from all available actions: IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER
    actions = ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]
    selected_action = random.choice(actions)
    action = env.unwrapped.action_type.actions_indexes[selected_action]
    return action
