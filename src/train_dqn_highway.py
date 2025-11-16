"""
Train and evaluate a DQN agent on the unmodified highway-v0 environment.

This script:
- Trains DQN on the default highway-v0 environment.
- Records total reward per episode.
- Saves a training reward plot under model/plots/.
- Evaluates the trained agent and prints collision rate and average speed.
"""

import os
import numpy as np
import gymnasium
import highway_env
from matplotlib import pyplot as plt

from performance_metrics import get_collision_rate, get_average_speed
from dqn_agent import DQNAgent


def flatten_observation(obs):
    """
    Convert environment observation to a 1D NumPy array.

    Highway-env may return arrays or dicts. Here we always flatten to a vector
    so that it can be used as the input to the DQN.
    """
    if isinstance(obs, dict):
        obs = obs.get("observation", obs)
    array = np.array(obs, dtype=np.float32)
    return array.flatten()


def train_dqn_on_highway(episodes=200, max_steps=50):
    """
    Train DQN on the default (unmodified) highway-v0 environment.

    Args:
        episodes: Number of training episodes.
        max_steps: Maximum steps per episode.

    Returns:
        env: The environment instance used for training.
        agent: Trained DQN agent.
        episode_rewards: List of total rewards per episode.
    """
    env = gymnasium.make("highway-v0", render_mode="rgb_array")
    obs, info = env.reset()
    state = flatten_observation(obs)
    state_dim = state.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        memory_capacity=50000,
        batch_size=64,
        target_update_interval=10,
    )

    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        state = flatten_observation(obs)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = flatten_observation(next_obs)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        print(
            f"Episode {episode + 1}/{episodes}, "
            f"Total reward: {total_reward:.2f}, "
            f"Last 10-episode average: {avg_reward:.2f}, "
            f"Epsilon: {agent.epsilon:.2f}"
        )

    return env, agent, episode_rewards


def evaluate_dqn(env, agent, episodes=50, max_steps=50):
    """
    Evaluate a trained DQN agent on highway-v0 and compute metrics.

    Args:
        env: Environment instance.
        agent: Trained DQN agent.
        episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.

    Returns:
        collision_rate: Collision rate in percentage.
        average_speed: Average speed across episodes.
    """
    total_collisions = 0
    total_epochs = 0
    episode_speeds = []

    # Use greedy policy for evaluation (epsilon = 0)
    epsilon_backup = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):
        obs, info = env.reset()
        state = flatten_observation(obs)
        step_speeds = []
        crashed = False

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, info = env.step(action)
            state = flatten_observation(next_obs)

            ego_speed = info.get("speed")
            if ego_speed is not None:
                step_speeds.append(ego_speed)

            if done or truncated:
                crashed = info.get("crashed", False)
                break

        episode_speeds.append(step_speeds)
        total_epochs += 1
        if crashed:
            total_collisions += 1

    # Restore epsilon after evaluation
    agent.epsilon = epsilon_backup

    collision_rate = get_collision_rate(total_collisions, total_epochs)
    average_speed = get_average_speed(episode_speeds)

    return collision_rate, average_speed


def save_reward_plot(episode_rewards, output_path):
    """
    Save the training reward curve to a PNG file.

    Args:
        episode_rewards: List of total rewards per episode.
        output_path: Path to the output image file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN on highway-v0: Training reward")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    # Train DQN on unmodified highway-v0
    env, agent, episode_rewards = train_dqn_on_highway(
        episodes=200,
        max_steps=50,
    )

    # Save convergence curve
    plots_dir = os.path.join("model", "plots")
    output_path = os.path.join(plots_dir, "dqn_highway_reward.png")
    save_reward_plot(episode_rewards, output_path)
    print(f"Saved training reward plot to: {output_path}")

    # Evaluate DQN and print driving performance metrics
    collision_rate, average_speed = evaluate_dqn(env, agent, episodes=50, max_steps=50)

    print("\n=== DQN Evaluation on unmodified highway-v0 ===")
    if collision_rate is not None:
        print(f"Collision rate: {collision_rate:.2f}%")
    else:
        print("Collision rate: None")

    if average_speed is not None:
        print(f"Average speed: {average_speed:.2f} (m/s)")
    else:
        print("Average speed: None")


if __name__ == "__main__":
    main()