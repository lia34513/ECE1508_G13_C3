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
│   ├── checkpoints/        # Saved model weights
│   ├── logs/              # Training logs and metrics
│   └── plots/             # Training visualizations
├── experiment/             # Experiment results
│   └── Ablation study results and performance comparisons
├── requirements.txt        # Python dependencies
└── README.md
```

## Usage

### Rule-based Method

The project currently includes a simple randome baseline:

- **Fixed speed & keep lane (IDLE action)**: Maintains the current lane and speed. Use this as a baseline for comparison with learned policies.

### Testing Environments

To test an environment, run the testing script:

```bash
python src/testing.py --env highway
```

Available options:
- `--env`: Specify the environment name (default: `highway`)
  - `highway`: Highway environment (`highway-v0`)
  - `roundabout`: Roundabout environment (`roundabout-v0`)
- `--render_mode`: Specify the render mode (default: `rgb_array`)
  - Common options: `rgb_array`, `human`
- `--steps`: Number of steps per epoch (default: `50`)
- `--epochs`: Number of epochs to run (default: `100`)
- `--method`: Rule-based method to use (default: `0`)
  - `0`: Random Policy - Fixed speed & keep lane
- `--high_speed_reward_weight`: High-speed reward weight (default: `1.0`)
- `--collision_reward_weight`: Collision reward weight (default: `-1.0`)

Examples:
```bash
# Test the highway environment with default settings (100 epochs, 50 steps per epoch, method 0)
python3 src/testing.py --env highway

# Full example with all options
python3 src/testing.py --env highway --render_mode human --epochs 100 --steps 50 --method 0 --high_speed_reward_weight 1 --collision_reward_weight -1.0
```


