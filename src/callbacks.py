"""
Callback utilities for training DQN and PPO models.
Provides shared callback creation functions for checkpointing and evaluation.
"""

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from typing import List, Union
import os
import time


class TimeEstimateCallback(BaseCallback):
    """
    Callback to track training time and estimate remaining time.
    """
    def __init__(self, total_timesteps: int, verbose: int = 1):
        super(TimeEstimateCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0
        self.num_timesteps_at_start = 0
        self.update_freq = 1000
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = self.num_timesteps
        self.num_timesteps_at_start = self.num_timesteps
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Update estimate periodically
        if self.num_timesteps - self.last_log_step >= self.update_freq:
            self._update_time_estimate()
            self.last_log_step = self.num_timesteps
            self.last_log_time = time.time()
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            elapsed_steps = self.num_timesteps - self.num_timesteps_at_start
            final_fps = elapsed_steps / total_time if total_time > 0 else 0
            self._print_time_info(total_time, is_final=True, fps=final_fps)
    
    def _update_time_estimate(self):
        """Update and print time estimate."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        elapsed_steps = self.num_timesteps - self.num_timesteps_at_start
        
        if elapsed_steps > 0 and elapsed_time > 0:
            fps = elapsed_steps / elapsed_time
            remaining_steps = self.total_timesteps - self.num_timesteps
            estimated_remaining = remaining_steps / fps if fps > 0 else 0
            
            self._print_time_info(elapsed_time, estimated_remaining, fps)
    
    def _print_time_info(self, elapsed_time, estimated_remaining=0, fps=0, is_final=False):
        """Print formatted time information."""
        def format_time(seconds):
            if seconds < 0:
                return "Unknown"
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        
        progress = (self.num_timesteps / self.total_timesteps * 100) if self.total_timesteps > 0 else 0
        
        if is_final:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {format_time(elapsed_time)}")
            print(f"Average FPS: {fps:.2f}")
            print(f"{'='*60}\n")
        else:
            print(f"\n[Time Estimate] Progress: {progress:.1f}% | "
                  f"Elapsed: {format_time(elapsed_time)} | "
                  f"Remaining: {format_time(estimated_remaining)} | "
                  f"FPS: {fps:.2f}")


def create_training_callbacks(
    model_type: str,
    eval_env: Union[object, None],
    checkpoint_dir: str,
    log_dir: str,
    eval_freq: int = int(1e4),
    n_eval_episodes: int = 10,
    total_timesteps: int = None,
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
        total_timesteps: Total training timesteps for time estimation (optional)
    
    Returns:
        List with callbacks: [time_callback, checkpoint_callback, eval_callback]
    """
    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine model-specific settings
    model_type_lower = model_type.lower()
    save_replay_buffer = model_type_lower == 'dqn'  # Only DQN uses replay buffer
    name_prefix = f'{model_type_lower}_checkpoint'
    
    # Create time estimate callback if total_timesteps is provided
    callbacks_list = []
    if total_timesteps is not None:
        time_callback = TimeEstimateCallback(total_timesteps=total_timesteps, verbose=1)
        callbacks_list.append(time_callback)
    
    # Create checkpoint callback - saves every eval_freq steps
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=checkpoint_dir,
        name_prefix=name_prefix,
        save_replay_buffer=save_replay_buffer,
        save_vecnormalize=True,
    )
    callbacks_list.append(checkpoint_callback)
    
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
    callbacks_list.append(eval_callback)
    
    return callbacks_list

