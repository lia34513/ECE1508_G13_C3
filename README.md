# ECE1508_G13_C3_RL for Autonomous Driving

Advancements in reinforcement learning (RL) and deep neural networks (DNNs) are tackling real-world challenges like robotic control, traffic signal management, and portfolio optimization. Autonomous driving is a standout application, demanding decision-making under uncertainty to balance safety, efficiency, and comfort.

We design and implement an RL agent for basic autonomous driving tasks such as lane-keeping, overtaking, or collision avoidance. The project aims to compare RL-based driving policies with simple rule-based baselines in lightweight simulators such as [Highway-env](https://github.com/Farama-Foundation/HighwayEnv/tree/master).

## Project Overview

We compare the following factors to perform ablation experiments:

* Rule-based baselines vs. RL agents
* Different scenarios: Highway
* Introduce modifications: Vehicles Density changes from 1.0 to 1.25 

We evaluate and analyze driving performance metrics, including collisions, average speed and average reward, while comparing RL agents against baselines under different conditions.

## Installation

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Project Structure

```
ECE1508_G13_C3/
├── src/                    # Source code
│   ├── Model structures and algorithms (RL agents, baselines, neural networks)
│   ├── Training and testing scripts
│   ├── Ablation experiment configurations
│   └── Configuration files and utilities
├── model/                  # PyTorch model artifacts
│   ├── DQN_buffer_size_25000/     # DQN with buffer size 25k
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_buffer_size_50000/    # DQN with buffer size 50k (default)
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_buffer_size_100000/   # DQN with buffer size 100k
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_gamma_0.99/           # DQN with gamma=0.99
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_linear_learning_rate/ # DQN with linear LR schedule
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_combine_lr_gamma/     # DQN with gamma=0.99 and linear LR
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── DQN_vehicles_density_1.25/ # DQN with traffic density 1.25
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── PPO_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1/  # PPO baseline
│   │   ├── checkpoints/         # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── PPO_vehicles_density_1.25_high_speed_reward_0.4_collision_reward_-1/  # PPO with density 1.25
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   ├── PPO_speed_0.6/            # PPO with speed reward 0.6
│   │   ├── checkpoints/          # Saved model weights and checkpoints
│   │   └── logs/                 # Training logs and metrics
│   └── PPO_speed_0.8/            # PPO with speed reward 0.8
│       ├── checkpoints/          # Saved model weights and checkpoints
│       └── logs/                 # Training logs and metrics
├── experiment/             # Experiment results
│   └── Ablation study results and performance comparisons
├── requirements.txt        # Python dependencies
└── README.md
```

## Usage

### Training DQN Agent

To train a DQN agent on the highway-v0 environment:

```bash
python3 src/training_dqn.py
```

**Default hyperparameters:**
- **Network architecture**: `[256, 256]` (2 hidden layers, 256 neurons each)
- **Learning rate**: `5e-4` (constant)
- **Replay buffer size**: `50,000`
- **Learning starts**: `500` (steps before training begins)
- **Batch size**: `32`
- **Gamma (discount factor)**: `0.95`
- **Train frequency**: `1` (train every step)
- **Gradient steps**: `1` (1 gradient update per training step)
- **Target network update interval**: `50` steps
- **Exploration fraction**: `0.5` (epsilon decays over 50% of training)
- **Exploration final epsilon**: `0.05` (minimum exploration rate)

Training configuration:
- **Total timesteps**: 600,000 (6e5)
- **Evaluation frequency**: Every 1,000 (1e3) steps
  - Model is evaluated and checkpoint is saved at these intervals
  - Evaluation runs 100 episodes per evaluation
- **Time estimation**: Training progress and time estimates are displayed automatically

Available options:
- `--n_envs`: Number of parallel environments (default: `1`)
  - Using multiple environments can speed up training but uses more memory
  - Example: `python3 src/training_dqn.py --n_envs 4`
- `--model_dir`: Custom directory name for saving models and logs (default: `DQN`)
  - Allows organizing multiple experiments with different directory names
  - Example: `python3 src/training_dqn.py --model_dir DQN_gamma0.99`
  - Files will be saved to `model/DQN_gamma0.99/checkpoints/` and `model/DQN_gamma0.99/logs/`

**Example with all options:**
```bash
python3 src/training_dqn.py --n_envs 8 --model_dir DQN_gamma0.99
```

### Training PPO Agent

To train a PPO agent on the highway-v0 environment:

```bash
python3 src/training_ppo.py
```

**Default hyperparameters:**
- **Network architecture**: `[256, 256]` (2 hidden layers, 256 neurons each)
- **Learning rate**: `5e-4` (constant)
- **Number of steps per update**: `1024` (per environment)
- **Batch size**: `64`
- **Number of epochs**: `10` (train on collected data for 10 epochs)
- **Gamma (discount factor)**: `0.95`
- **GAE lambda**: `0.95` (for advantage estimation)
- **Clip range**: `0.2` (PPO clipping parameter)
- **Entropy coefficient**: `0.01` (encourages exploration)
- **Value function coefficient**: `0.5` (value function loss weight)
- **Max gradient norm**: `0.5` (gradient clipping)

Training configuration:
- **Total timesteps**: 600,000 (6e5)
- **Evaluation frequency**: Every 1,000 (1e3) steps
  - Model is evaluated and checkpoint is saved at these intervals
  - Evaluation runs 100 episodes per evaluation
- **Time estimation**: Training progress and time estimates are displayed automatically

Available options:
- `--n_envs`: Number of parallel environments (default: `8`)
  - PPO is designed to work well with parallel environments
  - Using more environments can speed up training but uses more resources
  - Example: `python3 src/training_ppo.py --n_envs 16`
- `--model_dir`: Custom directory name for saving models and logs (default: `PPO`)
  - Allows organizing multiple experiments with different directory names
  - Example: `python3 src/training_ppo.py --model_dir PPO_experiment1`
  - Files will be saved to `model/PPO_experiment1/checkpoints/` and `model/PPO_experiment1/logs/`
- `--resume`: Path to checkpoint file to resume training from (default: `None`)
  - Allows continuing training from a saved checkpoint
  - Example: `python3 src/training_ppo.py --resume model/PPO/checkpoints/ppo_checkpoint_100000_steps.zip`
  - When resuming, training will continue for the specified total timesteps from the checkpoint

**Example with all options:**
```bash
# Start new training with custom directory
python3 src/training_ppo.py --n_envs 8 --model_dir PPO_baseline

# Resume training from checkpoint
python3 src/training_ppo.py --resume model/PPO/checkpoints/ppo_checkpoint_100000_steps.zip --model_dir PPO_resumed
```

### Viewing Training Logs with TensorBoard

Reward/loss curves and Stable-Baselines3 logs are written under `model/{model_dir}/logs/`.  
Launch TensorBoard in a separate terminal to explore them:

```bash
# For default DQN directory
tensorboard --logdir model/DQN/logs --port 6006

# For custom directory
tensorboard --logdir model/DQN_experiment1/logs --port 6006
```

Then open the URL (for example http://localhost:6006) in your browser.

### Testing Methods

The project supports three testing methods:

1. **Method 0**: OPD (Optimistic Planning)
   - Uses optimistic planning for decision-making
   - Configurable with `--opd_budget` and `--opd_gamma` parameters

2. **Method 1**: DQN agent (RL-based)
   - Uses a trained DQN agent for decision-making
   - Requires a trained checkpoint (default: `model/DQN_buffer_size_50000/checkpoints/best_model.zip`)
   - Automatically loads the checkpoint when method 1 is selected
   - The `best_model.zip` is saved automatically during training based on highest evaluation mean reward
   - If using custom model directory, use `--checkpoint_path` to specify the checkpoint file

3. **Method 2**: PPO agent (RL-based)
   - Uses a trained PPO agent for decision-making
   - Requires a trained checkpoint (default: `model/PPO_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1/best_model.zip`)
   - Automatically loads the checkpoint when method 2 is selected
   - The `best_model.zip` is saved automatically during training based on highest evaluation mean reward
   - If using custom model directory, use `--checkpoint_path` to specify the checkpoint file

### Testing Environments

To test an environment, run the testing script:

```bash
python3 src/testing.py --env highway
```

Available options:
- `--env`: Specify the environment name (default: `highway`)
  - `highway`: Highway environment (`highway-v0`)
  - `roundabout`: Roundabout environment (`roundabout-v0`)
- `--render_mode`: Specify the render mode (default: `rgb_array`)
  - Common options: `rgb_array`, `human`
- `--epochs`: Number of epochs to run (default: `100`)
- `--method`: Testing method to use (default: `0`)
  - `0`: OPD (Optimistic Planning)
  - `1`: DQN agent (requires trained checkpoint)
  - `2`: PPO agent (requires trained checkpoint)
- `--checkpoint_path`: Path to checkpoint file (for method 1 or 2)
  - If provided, uses the specified checkpoint file
  - If not provided, uses default paths:
    - Method 1 (DQN): `model/DQN_buffer_size_50000/checkpoints/best_model.zip`
    - Method 2 (PPO): `model/PPO_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1/best_model.zip`
  - Example: `--checkpoint_path model/DQN/checkpoints/best_model.zip`
- `--duration`: The duration of the episode (default: [env_config](src/env_config.py))
- `--high_speed_reward_weight`: High-speed reward weight (default: [env_config](src/env_config.py))
- `--collision_reward_weight`: Collision reward weight (default: [env_config](src/env_config.py))
- `--traffic_density`: The density of the traffic. 1.0 is the default, 1.25 is the high density (default: [env_config](src/env_config.py))
- `--opd_budget`: OPD planning budget (number of expansions, default: `50`, only for method 0)
- `--opd_gamma`: OPD discount factor (default: `0.7`, only for method 0)

**Note**: Environment configuration parameters (collision reward weight, high speed reward weight, traffic density, duration) default to values in `src/env_config.py` but can be overridden via command-line arguments.

### Examples

**Testing with OPD agent (Method 0):**
```bash
# Example of running OPD
python3 src/testing.py --env highway --method 0 --epochs 20 --render_mode rgb_array
```

**Testing with DQN agent (Method 1):**
```bash
# Using default checkpoint path
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 1

# Using custom checkpoint path
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 1 --checkpoint_path model/DQN_buffer_size_50000/checkpoints/best_model.zip
```

**Testing with PPO agent (Method 2):**
```bash
# Using default checkpoint path
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 2

# Using custom checkpoint path
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 2 --checkpoint_path model/PPO_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1/best_model.zip
```

**Testing with custom checkpoint:**
```bash
# Test with a specific checkpoint file
python3 src/testing.py --method 1 --checkpoint_path model/DQN_experiment1/checkpoints/dqn_checkpoint_100000_steps.zip --epochs 50
```