from stable_baselines3 import DQN
from env_config import get_highway_config
from callbacks import create_training_callbacks
import gymnasium
import highway_env
import os

# Project root directory: parent of src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = get_highway_config()

def train_dqn():
    # Create training environment
    env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
    
    # Create evaluation environment
    eval_env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
    
    # Set up directories (under project root: model/DQN/...)
    model_dir = os.path.join(BASE_DIR, "model", "DQN")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(model_dir, "logs")
    
    # Total training timesteps
    total_timesteps = int(2e4)
    
    # Calculate eval_freq as total_timesteps / 10 (evaluate and save checkpoint every 10% of training)
    eval_freq = max(1, int(total_timesteps / 10))
    
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
    
    print(f"Starting DQN training for {total_timesteps} timesteps with evaluation and checkpoint every {eval_freq} steps...")
    model.learn(total_timesteps, callback=callbacks)
    
    # Save final checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"dqn_highway_vehicles_density_{config['vehicles_density']}_high_speed_reward_{config['high_speed_reward']}_collision_reward_{config['collision_reward']}_final.zip")
    model.save(checkpoint_path)
    print(f"Saved final checkpoint to: {checkpoint_path}")
    
    # Close environments
    env.close()
    eval_env.close()

train_dqn()
