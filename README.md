# ECE1508_G13_C3_RL for Autonomous Driving

Advancements in reinforcement learning (RL) and deep neural networks (DNNs) are tackling real-world challenges like robotic control, traffic signal management, and portfolio optimization. Autonomous driving is a standout application, demanding decision-making under uncertainty to balance safety, efficiency, and comfort.

We design and implement an RL agent for basic autonomous driving tasks such as lane-keeping, overtaking, or collision avoidance. The project aims to compare RL-based driving policies with simple rule-based baselines in lightweight simulators such as [Highway-env](https://github.com/Farama-Foundation/HighwayEnv/tree/master).

## Project Overview

We compare the following factors to perform ablation experiments:

* Rule-based baselines vs. RL agents
* Different scenarios: Highway, Merge, Roundabout, and Parking
* Introduce modifications: Gaussian noise in sensor observations and delays/perturbations in actions

We evaluate and analyze driving performance metrics, including collisions and average speed, while comparing RL agents against baselines under different conditions.

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
│   ├── DQN/                # DQN model artifacts (or custom directory name)
│   │   ├── checkpoints/    # Saved model weights and checkpoints
│   │   └── logs/           # Training logs and metrics
│   └── PPO/                # PPO model artifacts (or custom directory name)
│       ├── checkpoints/    # Saved model weights and checkpoints
│       └── logs/           # Training logs and metrics
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

### Training Time Estimation

Both DQN and PPO training scripts now include automatic time estimation:
- **Real-time progress**: Shows training progress percentage
- **Elapsed time**: Displays how long training has been running
- **Estimated remaining time**: Calculates and displays estimated time to completion
- **FPS (Frames Per Second)**: Shows current training speed
- **Final summary**: Displays total training time and average FPS when training completes

Time estimates are updated every 1000 steps and provide accurate predictions based on current training speed.

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
   - Requires a trained checkpoint at `model/DQN/checkpoints/dqn_highway_vehicles_density_{density}_high_speed_reward_{reward}_collision_reward_{reward}.zip`
   - Automatically loads the checkpoint when method 1 is selected
   - If using custom model directory, ensure the checkpoint path matches

3. **Method 2**: PPO agent (RL-based)
   - Uses a trained PPO agent for decision-making
   - Requires a trained checkpoint at `model/PPO/checkpoints/ppo_highway_vehicles_density_{density}_high_speed_reward_{reward}_collision_reward_{reward}.zip`
   - Automatically loads the checkpoint when method 2 is selected

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
# First, ensure you have trained a DQN model
python3 src/training_dqn.py --model_dir DQN_baseline

# Then test with the trained model
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 1
```

**Testing with PPO agent (Method 2):**
```bash
# First, ensure you have trained a PPO model
python3 src/training_ppo.py --model_dir PPO_baseline

# Then test with the trained model
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 2
```