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
│   └── DQN/                # DQN model artifacts
│       ├── Checkpoint/     # Saved model weights and checkpoints
│       ├── Logs/           # Training logs and metrics
│       └── Plots/          # Training visualizations (reward and loss plots)
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

### Viewing Training Logs with TensorBoard

Reward/loss curves and Stable-Baselines3 logs are written under `model/DQN/Logs`.  
Launch TensorBoard in a separate terminal to explore them:

```bash
tensorboard --logdir model/DQN/Logs --port 6006
```

Then open the URL (for example http://localhost:6006) in your browser. 

### Testing Methods

The project supports two testing methods:

1. **Method 0**: OPD (Optimistic Planning)
   - Uses optimistic planning for decision-making
   - Configurable with `--opd_budget` and `--opd_gamma` parameters

2. **Method 1**: DQN agent (RL-based)
   - Uses a trained DQN agent for decision-making
   - Requires a trained checkpoint at `model/DQN/checkpoints/dqn_highway_vehicles_density_1_high_speed_reward_0.4_collision_reward_-1.zip`
   - Automatically loads the checkpoint when method 1 is selected

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
python3 src/training_dqn.py

# Then test with the trained model
python3 src/testing.py --env highway --render_mode human --epochs 100 --method 1
```