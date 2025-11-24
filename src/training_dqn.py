from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
    """
    def _init():
        env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
        env.reset(seed=seed)
        return env
    return _init


def train_dqn(n_envs: int = 1):
    """
    Train DQN model.
    
    Args:
        n_envs: Number of parallel environments (default: 1)
    """
    # Create training environment(s)
    if n_envs > 1:
        # Use vectorized environment for multiple parallel envs
        env = DummyVecEnv([make_env_fn(seed=i) for i in range(n_envs)])
    else:
        # Use single environment
        env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
    
    # Create evaluation environment with same seed as testing.py for consistency
    # This ensures evaluation during training uses the same scenarios
    eval_env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
    eval_env.reset(seed=1000)
    
    # Set up directories (under project root: model/DQN/...)
    model_dir = os.path.join(BASE_DIR, "model", "DQN")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(model_dir, "logs")
    
    # Total training timesteps
    total_timesteps = int(2e6)
    
    
    # Evaluation frequency (default from callbacks.py)
    eval_freq = int(1e4)  # 10,000 steps
    
    # Create callbacks using shared function - evaluates and saves every eval_freq steps
    callbacks = create_training_callbacks(
        model_type='dqn',
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )
    
    model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    exploration_fraction=0.7,
                    verbose=1,
                    tensorboard_log=os.path.join(log_dir, f"vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}"))
    
    # Check and report device being used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    env_info = f"with {n_envs} parallel environment(s)" if n_envs > 1 else "with single environment"
    print(f"Starting DQN training for {total_timesteps} timesteps {env_info}...")
    print(f"Evaluation and checkpoint every {eval_freq} steps...")
    model.learn(total_timesteps, callback=callbacks)
    
    # Save final checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"dqn_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}_final.zip")
    model.save(checkpoint_path)
    print(f"Saved final checkpoint to: {checkpoint_path}")
    
    # Close environments
    env.close()
    eval_env.close()


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train DQN model for highway-v0 environment")
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)"
    )
    
    args = parser.parse_args()
    train_dqn(n_envs=args.n_envs)


if __name__ == "__main__":
    main()
