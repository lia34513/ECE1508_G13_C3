# Model Artifacts

This directory contains trained model checkpoints and training logs for different hyperparameter configurations.

## DQN Models

### DQN_buffer_size_50000 (Default)
- **Network**: `[256, 256]`
- **Learning rate**: `5e-4` (constant)
- **Buffer size**: `50,000`
- **Learning starts**: `500`
- **Batch size**: `32`
- **Gamma**: `0.95`
- **Train frequency**: `1`
- **Gradient steps**: `1`
- **Target update interval**: `50`
- **Exploration fraction**: `0.5`
- **Total timesteps**: `600,000`
- **Environment**: `vehicles_density=1.0`, `high_speed_reward=0.4`, `collision_reward=-1`

### DQN_buffer_size_25000
- Same as default, except:
- **Buffer size**: `25,000`

### DQN_buffer_size_100000
- Same as default, except:
- **Buffer size**: `100,000`

### DQN_gamma_0.99
- Same as default, except:
- **Gamma**: `0.99`

### DQN_linear_learning_rate
- Same as default, except:
- **Learning rate**: Linear schedule from `5e-4` to `1e-5`

### DQN_combine_lr_gamma
- Same as default, except:
- **Gamma**: `0.99`
- **Learning rate**: Linear schedule from `5e-4` to `1e-5`

### DQN_vehicles_density_1.25
- Same as default, except:
- **Environment**: `vehicles_density=1.25`

## PPO Models

### PPO_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1 (Default)
- **Network**: `[256, 256]`
- **Learning rate**: `5e-4` (constant)
- **n_steps**: `1024`
- **Batch size**: `64`
- **n_epochs**: `10`
- **Gamma**: `0.95`
- **GAE lambda**: `0.95`
- **Clip range**: `0.2`
- **Entropy coefficient**: `0.01`
- **Value function coefficient**: `0.5`
- **Max gradient norm**: `0.5`
- **Total timesteps**: `600,000`
- **Environment**: `vehicles_density=1.0`, `high_speed_reward=0.4`, `collision_reward=-1`

### PPO_vehicles_density_1.25_high_speed_reward_0.4_collision_reward_-1
- Same as default, except:
- **Environment**: `vehicles_density=1.25`

### PPO_speed_0.6
- Same as default, except:
- **Environment**: `high_speed_reward=0.6`

### PPO_speed_0.8
- Same as default, except:
- **Environment**: `high_speed_reward=0.8`

## Common Training Settings

- **Evaluation frequency**: Every `1,000` steps
- **Evaluation episodes**: `100` episodes per evaluation
- **Best model**: Saved based on highest evaluation mean reward
