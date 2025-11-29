from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from env_config import get_highway_config
from callbacks import create_training_callbacks
import gymnasium
import highway_env
import os
import argparse
import torch

# Project root directory: parent of src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = get_highway_config()


def make_env_fn(seed: int):
    """
    Factory function to create a single highway-v0 environment.
    Wrapped with Monitor to track episode statistics (ep_len_mean, ep_rew_mean).
    """

    def _init():
        env = gymnasium.make("highway-v0", config=config, render_mode="rgb_array")
        # Wrap with Monitor to enable episode statistics tracking
        env = Monitor(env)
        return env

    return _init


def train_dqn(n_envs: int = 1, model_dir_name: str = "DQN"):
    """
    Train DQN model.

    Args:
        n_envs: Number of parallel environments (default: 1)
        model_dir_name: Name of the directory to save models and logs (default: "DQN")
    """
    # Create training environment(s)
    if n_envs > 1:
        # Use vectorized environment for multiple parallel envs
        env = DummyVecEnv([make_env_fn(seed=i) for i in range(n_envs)])
    else:
        # Use single environment with Monitor wrapper for episode statistics
        env = gymnasium.make("highway-v0", config=config, render_mode="rgb_array")
        env = Monitor(env)

    # Create evaluation environment with same seed as testing.py for consistency
    # This ensures evaluation during training uses the same scenarios
    # Wrap with Monitor to track evaluation episode statistics
    eval_env = gymnasium.make("highway-v0", config=config, render_mode="rgb_array")
    eval_env.reset(seed=1000)
    eval_env = Monitor(eval_env)

    # Set up directories (under project root: model/{model_dir_name}/...)
    model_dir = os.path.join(BASE_DIR, "model", model_dir_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(model_dir, "logs")

    # Total training timesteps
    total_timesteps = int(6e5)

    # Evaluation frequency (default from callbacks.py)
    eval_freq = int(1e3)  # 1,000 steps

    # Create callbacks using shared function - evaluates and saves every eval_freq steps
    callbacks = create_training_callbacks(
        model_type="dqn",
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=100,
        total_timesteps=total_timesteps,
    )

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=25000,
        learning_starts=500,
        batch_size=32,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.5,
        verbose=1,
        tensorboard_log=os.path.join(
            log_dir,
            f"vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}",
        ),
    )

    # Check and report device being used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print(f"Model directory: {model_dir}")
    env_info = (
        f"with {n_envs} parallel environment(s)"
        if n_envs > 1
        else "with single environment"
    )
    print(f"Starting DQN training for {total_timesteps} timesteps {env_info}...")
    print(f"Evaluation and checkpoint every {eval_freq} steps...")

    # Continue training with callbacks
    model.learn(total_timesteps, callback=callbacks)

    # Save final checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"dqn_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}_final.zip",
    )
    model.save(checkpoint_path)
    print(f"Saved final checkpoint to: {checkpoint_path}")

    # Close environments
    env.close()
    eval_env.close()


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train DQN model for highway-v0 environment"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="DQN",
        help="Name of the directory to save models and logs (default: 'DQN')",
    )

    args = parser.parse_args()
    train_dqn(n_envs=args.n_envs, model_dir_name=args.model_dir)


if __name__ == "__main__":
    main()
