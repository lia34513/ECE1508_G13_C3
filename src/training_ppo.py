from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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
        env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
        # Wrap with Monitor to enable episode statistics tracking
        env = Monitor(env)
        return env
    return _init


def train_ppo(n_envs: int = 8):
    """
    Train PPO model.
    
    Args:
        n_envs: Number of parallel environments (default: 8)
    """

   
    # Create training environment(s)
    if n_envs > 1:
        # Use vectorized environment for multiple parallel envs
        env = DummyVecEnv([make_env_fn(seed=i) for i in range(n_envs)])
    else:
        # Use single environment with Monitor wrapper for episode statistics
        env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
        env = Monitor(env)

    # log and checkpoint directories (under project root: model/PPO/...)
    model_dir = os.path.join(BASE_DIR, "model", "PPO")
    log_dir = os.path.join(model_dir, "logs")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    # create evaluation environment (single env for evaluation)
    # Use same seed for consistency
    # Wrap with Monitor to track evaluation episode statistics
    eval_env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
    eval_env.reset(seed=1000)
    eval_env = Monitor(eval_env)

    # total timesteps is counted across all envs
    total_timesteps = int(2e6)
    
    # Evaluation frequency (default from callbacks.py)
    eval_freq = int(1e3)  # 1,000 steps

    # Create callbacks using shared function - evaluates and saves every eval_freq steps
    callbacks = create_training_callbacks(
        model_type='ppo',
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=100,
    )

    # initialize PPO model
    # NOTE: n_steps is per-env, so total rollout size per update is n_envs * n_steps.
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        n_steps=1024,      # per env → total batch = n_envs * n_steps
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Check and report device being used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(
        f"Start parallel PPO training for {total_timesteps} timesteps "
        f"with n_envs={n_envs}..."
    )
    print(f"Evaluation and checkpoint every {eval_freq} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    print("PPO training finished.")

    # save final checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"ppo_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}_final.zip"
    )
    model.save(checkpoint_path)
    print(f"Saved final PPO checkpoint to: {checkpoint_path}")

    # close environments
    env.close()
    eval_env.close()


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train PPO model for highway-v0 environment")
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)"
    )
    
    args = parser.parse_args()
    train_ppo(n_envs=args.n_envs)


if __name__ == "__main__":
    main()
