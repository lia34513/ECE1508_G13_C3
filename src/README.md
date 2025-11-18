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

## Hyperparameter Analysis: High-Speed Reward

To study the trade-off between **driving speed** and **safety** in the DQN highway-driving environment, we evaluated three configurations for the `high_speed_reward` parameter: **0.1**, **0.4**, and **0.7**.  
Each model was trained with the same settings and tested across 100 episodes.

### Results Summary

| High-Speed Reward | Avg Speed (m/s) | Collision Rate | Avg Reward | Interpretation |
|------------------|-----------------|----------------|------------|----------------|
| **0.1**          | 20.78           | **2%**         | **0.91**   | Prioritizes safety; maintains reasonable speed; *best overall balance*. |
| **0.4**          | 25.32           | 48%            | 0.84       | Rewards speed too strongly; agent becomes aggressive; collision risk significantly increases. |
| **0.7**          | 28.07           | 85%            | 0.89       | Agent maximizes speed at the cost of safety; **unsafe** for real deployment. |

### Interpretation

- **0.1 → “Safe / Conservative policy”**  
  - Collision rate only **2%**, the lowest among all settings.  
  - The agent drives safely while still achieving reasonably high speed.  
  - **Recommended setting** for environments where safety is the primary objective.

- **0.4 → “Risky policy”**  
  - Speed increases moderately, but collision rate jumps to **48%**.  
  - The agent becomes noticeably more aggressive.  
  - Not suitable if a safety threshold (e.g., <10% collision) is required.

- **0.7 → “Unsafe policy (Speed-maximizing)”**  
  - Highest driving speed, but collision rate reaches **85%**.  
  - The agent ignores safety almost entirely and focuses on maximizing velocity.  
  - **Not deployable** in any realistic or safety-critical setting.

### Conclusion

The results indicate a strong trade-off between speed and safety:

- When `high_speed_reward` is too large, the agent over-optimizes for speed and ignores safety.
- A small value (e.g., **0.1**) provides the best overall performance, achieving:
  - High reward  
  - Low collision rate  
  - Reasonable speed  

**Therefore, `high_speed_reward = 0.1` is selected as the optimal balance between safety and efficiency.**