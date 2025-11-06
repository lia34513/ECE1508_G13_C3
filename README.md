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

### Rule-based Methods

The project includes rule-based baseline methods for comparison:

- **Method 0 (fixed_speed_keep_lane)**: Maintains current lane and speed using IDLE action. This is the simplest baseline that does not change lanes or speed.

- **Method 1 (fixed_speed_random_lane)**: Maintains fixed speed but randomly changes lanes. At each step, it randomly chooses between staying in lane (IDLE), changing to the left lane, or changing to the right lane.

- **Method 2 (random_speed_keep_lane)**: Randomly changes speed while maintaining current lane. At each step, it randomly chooses between maintaining speed (IDLE), accelerating (FASTER), or decelerating (SLOWER).

- **Method 3 (random_speed_random_lane)**: Randomly changes both speed and lane. At each step, it randomly chooses from all available actions: IDLE, LANE_LEFT, LANE_RIGHT, FASTER, or SLOWER.

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
  - `0`: Fixed speed & keep lane
  - `1`: Fixed speed & change lane randomly
  - `2`: Random speed & keep lane
  - `3`: Random speed & random lane

Examples:
```bash
# Test the highway environment with default settings (100 epochs, 50 steps per epoch, method 0)
python src/testing.py --env highway

# Full example with all options
python src/testing.py --env highway --render_mode human --epochs 100 --steps 50 --method 0
```


