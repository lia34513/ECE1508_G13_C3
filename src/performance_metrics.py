"""
Performance metrics for testing autonomous driving policies.
"""

from typing import List, Optional


def get_collision_rate(total_collisions: int, total_epochs: int) -> Optional[float]:
    """Compute the collision rate (percentage) given collisions and epochs.

    Args:
        total_collisions: Number of epochs where a collision occurred.
        total_epochs: Total number of epochs evaluated.

    Returns:
        Collision rate as a percentage (float) if total_epochs > 0, otherwise None.
    """
    if total_epochs <= 0:
        return None
    return (total_collisions / total_epochs) * 100.0


def get_collision_rate_per_action(total_collisions: int, total_actions: int) -> Optional[float]:
    """Compute the collision rate per action (percentage) given collisions and actions.

    Args:
        total_collisions: Number of epochs where a collision occurred.
        total_actions: Total number of actions taken.

    Returns:
        Collision rate per action as a percentage (float) if total_actions > 0, otherwise None.
    """
    if total_actions <= 0:
        return None
    return (total_collisions / total_actions) * 100.0


def get_average_speed(episode_speeds: List[List[float]]) -> Optional[float]:
    """Compute average speed across episodes.

    The average speed is defined as the sum of per-episode mean speeds divided by the
    number of episodes. Each episode's mean speed is the sum of vehicle speeds at each
    step divided by the number of steps in that episode.

    Args:
        episode_speeds: A list of episodes where each episode is a list of speeds
            recorded at each step.

    Returns:
        Average speed across all episodes if available, otherwise None.
    """
    total_mean_speed = 0.0
    episode_count = 0

    for speeds in episode_speeds:
        if speeds is None:
            continue
        episode_mean = sum(speeds) / len(speeds)
        total_mean_speed += episode_mean
        episode_count += 1

    if episode_count == 0:
        return None

    return total_mean_speed / episode_count
