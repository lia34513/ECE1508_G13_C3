"""
Callback utilities for training DQN and PPO models.
Provides shared callback creation functions for checkpointing and evaluation.
"""

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from typing import List, Union
import os


def create_training_callbacks(
    model_type: str,
    eval_env: Union[object, None],
    checkpoint_dir: str,
    log_dir: str,
    eval_freq: int = int(1e4),
    n_eval_episodes: int = 10,
) -> List:
    """
    Create checkpoint and evaluation callbacks for training.
    
    Args:
        model_type: Type of model ('dqn' or 'ppo')
        eval_env: Evaluation environment (can be a gymnasium env or vec env)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        eval_freq: Frequency (in steps) to evaluate and save checkpoints
        n_eval_episodes: Number of episodes to run during evaluation
    
    Returns:
        List with callbacks: [checkpoint_callback, eval_callback]
    """
    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine model-specific settings
    model_type_lower = model_type.lower()
    save_replay_buffer = model_type_lower == 'dqn'  # Only DQN uses replay buffer
    name_prefix = f'{model_type_lower}_checkpoint'
    
    # Create checkpoint callback - saves every eval_freq steps
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=checkpoint_dir,
        name_prefix=name_prefix,
        save_replay_buffer=save_replay_buffer,
        save_vecnormalize=True,
    )
    
    # Create evaluation callback - evaluates every eval_freq steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir, 
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
    )
    
    return [checkpoint_callback, eval_callback]

