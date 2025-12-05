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
        env = Monitor(env)
        return env

    return _init


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate (default: 0.0)
    
    Returns:
        A function that takes progress_remaining (0 to 1) and returns learning rate
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will go from 1 (beginning) to 0 (end).
        """
        return final_value + (initial_value - final_value) * progress_remaining
    return func

def train_dqn(n_envs: int = 1, model_dir_name: str = "DQN", resume_from: str = None):
    """
    Train DQN model.

    Args:
        n_envs: Number of parallel environments (default: 1)
        model_dir_name: Name of the directory to save models and logs (default: "DQN")
        resume_from: Path to checkpoint to resume training from (default: None)
    """
    # Create training environment(s)
    if n_envs > 1:
        # Use vectorized environment for multiple parallel envs
        env = DummyVecEnv([make_env_fn(seed=i) for i in range(n_envs)])
    else:
        env = gymnasium.make("highway-v0", config=config, render_mode="rgb_array")
        env = Monitor(env)

    eval_env = gymnasium.make("highway-v0", config=config, render_mode="rgb_array")
    eval_env.reset(seed=1000)
    eval_env = Monitor(eval_env)

    model_dir = os.path.join(BASE_DIR, "model", model_dir_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(model_dir, "logs")

    total_timesteps = int(6e5)
    eval_freq = int(1e3)

    callbacks = create_training_callbacks(
        model_type="dqn",
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=100,
        total_timesteps=total_timesteps,
    )
    
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from}")
        model = DQN.load(resume_from, env=env)
        model.tensorboard_log = os.path.join(log_dir, f"vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}")
        print(f"Successfully loaded checkpoint!")
        print(f"Current timesteps completed: {model.num_timesteps}")
        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            print(f"Warning: Model has already completed {model.num_timesteps} timesteps (target: {total_timesteps})")
            print("No additional training needed.")
            env.close()
            eval_env.close()
            return
        print(f"Will train for {remaining_timesteps} more timesteps to reach {total_timesteps} total.")
    else:
        lr_schedule = linear_schedule(initial_value=5e-4, final_value=1e-5)
        
        model = DQN('MlpPolicy', env,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=5e-4,
                        buffer_size=50000,
                        learning_starts=500,
                        batch_size=32,
                        gamma=0.95,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        exploration_fraction=0.5,
                        verbose=1,
                        tensorboard_log=os.path.join(log_dir, f"vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}"))
        remaining_timesteps = total_timesteps
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print(f"Model directory: {model_dir}")
    env_info = f"with {n_envs} parallel environment(s)" if n_envs > 1 else "with single environment"
    print(f"Starting DQN training for {remaining_timesteps} timesteps {env_info}...")
    print(f"Evaluation and checkpoint every {eval_freq} steps...")

    model.learn(remaining_timesteps, callback=callbacks) 
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"dqn_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}_final.zip",
    )
    model.save(checkpoint_path)
    print(f"Saved final checkpoint to: {checkpoint_path}")

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
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., model/DQN/checkpoints/rl_model_495000_steps.zip)"
    )
    
    args = parser.parse_args()
    train_dqn(n_envs=args.n_envs, model_dir_name=args.model_dir, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
