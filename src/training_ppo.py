from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_config import get_highway_config
import gymnasium
import highway_env  # noqa: F401  # needed to register env
import os

# Project root directory: parent of src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = get_highway_config()


def make_env_fn(seed: int):
    """
    Factory function to create a single highway-v0 environment.
    Used by SubprocVecEnv to create multiple parallel environments.
    """
    def _init():
        env = gymnasium.make("highway-v0", config=config)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo():
    # number of parallel environments
    n_envs = 8

    # create vectorized environment with multiple processes
    env = SubprocVecEnv([make_env_fn(seed=i) for i in range(n_envs)])

    # log and checkpoint directories (under project root: model/PPO/...)
    model_dir = os.path.join(BASE_DIR, "model", "PPO")
    log_dir = os.path.join(model_dir, "logs")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    exp_suffix = (
        f"vehden_{config.get('vehicles_density', 'NA')}_"
        f"dur_{config.get('duration', 'NA')}_"
        f"nenvs_{n_envs}"
    )

    # initialize PPO model
    # NOTE: n_steps is per-env, so total rollout size per update is n_envs * n_steps.
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        n_steps=2048,      # per env → total batch = n_envs * n_steps
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

    # total timesteps is counted across all envs
    total_timesteps = int(2e4)

    print(
        f"Start parallel PPO training for {total_timesteps} timesteps "
        f"with n_envs={n_envs}..."
    )
    model.learn(total_timesteps=total_timesteps)
    print("PPO training finished.")

    # save checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir, f"ppo_highway_{exp_suffix}.zip"
    )
    model.save(checkpoint_path)
    print(f"Saved PPO checkpoint to: {checkpoint_path}")

    # close vectorized env
    env.close()


if __name__ == "__main__":
    train_ppo()
