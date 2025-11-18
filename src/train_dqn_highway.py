"""
Train a DQN agent on the highway-v0 environment.

This script:
- Trains DQN on the default highway-v0 environment.
- Records average reward and loss per episode.
- Saves training reward and loss plots under model/DQN/Plots/.
"""

import os
import numpy as np
import gymnasium
import highway_env
from matplotlib import pyplot as plt
import torch

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
        episode_losses: List of average loss per episode.
    """
    env = gymnasium.make("highway-v0", render_mode="rgb_array", config={"collision_reward": -1.0, "high_speed_reward": 0.4, "right_lane_reward": 0.0})
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
    episode_losses = []

    for episode in range(episodes):
        obs, info = env.reset()
        state = flatten_observation(obs)
        total_reward = 0.0
        episode_loss_values = []
        num_steps = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = flatten_observation(next_obs)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            loss = agent.train_step()
            
            if loss is not None:
                episode_loss_values.append(loss)

            state = next_state
            total_reward += reward
            num_steps += 1

            if done or truncated:
                break

        
        avg_loss = np.mean(episode_loss_values) if episode_loss_values else 0.0
        episode_losses.append(avg_loss)
        avg_reward = total_reward / num_steps if num_steps > 0 else 0.0
        episode_rewards.append(avg_reward)
        
        # Print average loss and average reward
        print(
            f"Episode {episode + 1}/{episodes}, "
            f"Average reward: {avg_reward:.4f}, "
            f"Average loss: {avg_loss:.4f}, "
            f"Epsilon: {agent.epsilon:.2f}"
        )

    return env, agent, episode_rewards, episode_losses


def save_reward_plot(episode_rewards, output_path):
    """
    Save the training reward curve to a PNG file showing individual values and moving average.

    Args:
        episode_rewards: List of total rewards per episode.
        output_path: Path to the output image file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    episodes = range(1, len(episode_rewards) + 1)
    
    # Calculate moving average (window size of 100, or smaller if not enough data)
    window_size = min(100, len(episode_rewards))
    moving_avg = []
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
    
    plt.figure(figsize=(10, 6))
    # Plot individual rewards as bars (grey)
    plt.bar(episodes, episode_rewards, alpha=0.3, color='grey', label='Reward', width=1.0)
    # Plot moving average as line (red)
    plt.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on highway-v0: Training reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_plot(episode_losses, output_path):
    """
    Save the training loss curve to a PNG file showing individual values and moving average.

    Args:
        episode_losses: List of average loss per episode.
        output_path: Path to the output image file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    episodes = range(1, len(episode_losses) + 1)
    
    # Calculate moving average (window size of 100, or smaller if not enough data)
    window_size = min(100, len(episode_losses))
    moving_avg = []
    for i in range(len(episode_losses)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(episode_losses[start_idx:i+1]))
    
    plt.figure(figsize=(10, 6))
    # Plot individual losses as bars (grey)
    plt.bar(episodes, episode_losses, alpha=0.3, color='grey', label='Loss', width=1.0)
    # Plot moving average as line (red)
    plt.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Average')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DQN on highway-v0: Training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_checkpoint(agent, checkpoint_path):
    """
    Save DQN agent checkpoint.

    Args:
        agent: Trained DQN agent.
        checkpoint_path: Path to save the checkpoint file.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
    }, checkpoint_path)


def main():
    # Train DQN on unmodified highway-v0
    env, agent, episode_rewards, episode_losses = train_dqn_on_highway(
        episodes=200,
        max_steps=50,
    )

    # Save convergence curves
    plots_dir = os.path.join("model", "DQN", "Plots")
    reward_output_path = os.path.join(plots_dir, "dqn_highway_reward.png")
    save_reward_plot(episode_rewards, reward_output_path)
    print(f"Saved training reward plot to: {reward_output_path}")

    # Save training loss plot
    loss_output_path = os.path.join(plots_dir, "dqn_highway_loss.png")
    save_loss_plot(episode_losses, loss_output_path)
    print(f"Saved training loss plot to: {loss_output_path}")

    # Save checkpoint
    checkpoint_dir = os.path.join("model", "DQN", "Checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "dqn_highway.pth")
    save_checkpoint(agent, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()