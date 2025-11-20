## OPD
### Unmodified Environment (Default Settings)
```bash
>>> python3 src/testing.py --env highway --method 1 --epochs 20 --duration 40 --render_mode rgb_array

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 20
Method: OPD (Optimistic Planning)
Traffic Density: 1.0
High Speed Reward Weight: 1.0
Collision Reward Weight: -1.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 2/20: OK, Total reward: 35.94, Avg reward: 0.90
Epoch 3/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 4/20: OK, Total reward: 39.94, Avg reward: 1.00
Epoch 5/20: OK, Total reward: 39.94, Avg reward: 1.00
Epoch 6/20: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 7/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 8/20: OK, Total reward: 37.19, Avg reward: 0.93
Epoch 9/20: OK, Total reward: 36.44, Avg reward: 0.91
Epoch 10/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 11/20: OK, Total reward: 39.94, Avg reward: 1.00
Epoch 12/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 13/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 14/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 15/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 16/20: OK, Total reward: 39.94, Avg reward: 1.00
Epoch 17/20: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 18/20: OK, Total reward: 39.94, Avg reward: 1.00
Epoch 19/20: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 20/20: OK, Total reward: 39.93, Avg reward: 1.00

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 20
Collisions: 0
Collision rate: 0.00%
Average speed: 29.72 m/s
Average total reward: 39.42

=== Summary for Table ===
Collision Rate: 0.00%
Average Speed: 29.72 m/s
Average Reward: 39.42
```

### Modified Environments (Different Reward Weights)
High Speed Reward Weight:
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --high_speed_reward_weight 2.0

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 100
Method: OPD (Optimistic Planning)
Traffic Density: 1.0
High Speed Reward Weight: 2.0
Collision Reward Weight: -1.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 2/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 3/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 4/100: OK, Total reward: 38.91, Avg reward: 0.97
Epoch 5/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 6/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 7/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 8/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 9/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 10/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 11/100: OK, Total reward: 31.25, Avg reward: 0.78
Epoch 12/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 13/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 14/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 15/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 16/100: OK, Total reward: 38.05, Avg reward: 0.95
Epoch 17/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 18/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 19/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 20/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 21/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 22/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 23/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 24/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 25/100: OK, Total reward: 33.58, Avg reward: 0.84
Epoch 26/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 27/100: OK, Total reward: 39.90, Avg reward: 1.00
Epoch 28/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 29/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 30/100: OK, Total reward: 38.00, Avg reward: 0.95
Epoch 31/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 32/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 33/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 34/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 35/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 36/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 37/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 38/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 39/100: OK, Total reward: 39.91, Avg reward: 1.00
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Epoch 40/100: CRASHED, Total reward: 29.30, Avg reward: 0.98
Epoch 41/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 42/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 43/100: OK, Total reward: 33.73, Avg reward: 0.84
Epoch 44/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 45/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 46/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 47/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 48/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 49/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 50/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 51/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 52/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 53/100: OK, Total reward: 33.58, Avg reward: 0.84
Epoch 54/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 55/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 56/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 57/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 58/100: OK, Total reward: 37.25, Avg reward: 0.93
Epoch 59/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 60/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 61/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 62/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 63/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 64/100: OK, Total reward: 39.59, Avg reward: 0.99
Epoch 65/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 66/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 67/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 68/100: OK, Total reward: 39.93, Avg reward: 1.00
Epoch 69/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 70/100: OK, Total reward: 35.58, Avg reward: 0.89
Epoch 71/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 72/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 73/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 74/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 75/100: OK, Total reward: 39.90, Avg reward: 1.00
Epoch 76/100: OK, Total reward: 39.64, Avg reward: 0.99
Epoch 77/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 78/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 79/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 80/100: OK, Total reward: 38.57, Avg reward: 0.96
Epoch 81/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 82/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 83/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 84/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 85/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 86/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 87/100: OK, Total reward: 39.25, Avg reward: 0.98
Epoch 88/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 89/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 90/100: OK, Total reward: 39.24, Avg reward: 0.98
Epoch 91/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 92/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 93/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 94/100: OK, Total reward: 39.58, Avg reward: 0.99
Epoch 95/100: OK, Total reward: 37.58, Avg reward: 0.94
Epoch 96/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 97/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 98/100: OK, Total reward: 39.91, Avg reward: 1.00
Epoch 99/100: OK, Total reward: 39.92, Avg reward: 1.00
Epoch 100/100: OK, Total reward: 39.91, Avg reward: 1.00

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 100
Collisions: 1
Collision rate: 1.00%
Average speed: 29.80 m/s
Average total reward: 39.35

=== Summary for Table ===
Collision Rate: 1.00%
Average Speed: 29.80 m/s
Average Reward: 39.35
```

Higher Collision Penalty:
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --collision_reward_weight -5.0
```

Combined (High Speed + High Penalty):
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --high_speed_reward_weight 2.0 --collision_reward_weight -5.0
```

### High Traffic Density
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --traffic_density 1.25
```

### Compare with Fixed Speed Keep Lane Baseline
```bash
python3 src/testing.py --env highway --method 0 --epochs 100 --duration 40 --render_mode rgb_array
```

### OPD Parameter Tuning
Higher budget (more planning)
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --opd_budget 100
```

Different discount factor
```bash
python3 src/testing.py --env highway --method 1 --epochs 100 --duration 40 --render_mode rgb_array --opd_gamma 0.9
```