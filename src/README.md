# Source Code

This directory contains the source code for the project:

## Files

- **`testing.py`**: Main testing script for evaluating rule-based methods and RL agents in different environments
  - Supports multiple environments (highway, roundabout)
  - Configurable epochs, steps, and render modes
  - Tracks collision rates and performance metrics
  
- **`rule_based.py`**: Rule-based baseline policies for autonomous driving
  - `fixed_speed_keep_lane()`: Maintains current lane and speed (IDLE)
  - `fixed_speed_random_lane()`: Fixed speed with random lane changes
  - `random_speed_keep_lane()`: Random speed changes while keeping lane
  - `random_speed_random_lane()`: Random speed and random lane changes

## Future Additions

- Model structures and algorithms (RL agents, neural network architectures)
- Training scripts
- Ablation experiment configurations and runners
- Configuration files (hyperparameters, environment settings)
- Visualization and analysis utilities

