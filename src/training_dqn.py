from stable_baselines3 import DQN
from env_config import get_highway_config
import gymnasium
import highway_env

config = get_highway_config()

def train_dqn():
    env = gymnasium.make('highway-v0', config=config, render_mode='rgb_array')
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
                    tensorboard_log=f'model/DQN/Logs/vehicles_density_{config["vehicles_density"]}_high_speed_reward_{config["high_speed_reward"]}_collision_reward_{config["collision_reward"]}')
    model.learn(int(2e4))

train_dqn()
