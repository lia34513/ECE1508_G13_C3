# Source Code

This directory contains the source code for the project:

## Files

- **`testing.py`**: Main testing script for evaluating the rule-based baseline and future RL agents in different environments
  - Supports multiple environments (highway, roundabout)
  - Configurable epochs, steps, and render modes
  - Tracks collision rates and performance metrics
  
- **`rule_based.py`**: Rule-based baseline policy for autonomous driving
  - `fixed_speed_keep_lane()`: Maintains current lane and speed (IDLE)

## Future Additions

- Model structures and algorithms (RL agents, neural network architectures)
- Training scripts
- Ablation experiment configurations and runners
- Configuration files (hyperparameters, environment settings)
- Visualization and analysis utilities

