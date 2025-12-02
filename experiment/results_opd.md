## OPD
### 11.24
```
# from lab computer
Eval num_timesteps=400000, episode_reward=28.60 +/- 0.00
Episode length: 40.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 40          |
|    mean_reward          | 28.6        |
| time/                   |             |
|    total_timesteps      | 400000      |
| train/                  |             |
|    approx_kl            | 0.009995634 |
|    clip_fraction        | 0.105       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.984      |
|    explained_variance   | 0.456       |
|    learning_rate        | 0.0005      |
|    loss                 | 14.4        |
|    n_updates            | 240         |
|    policy_gradient_loss | 0.000362    |
|    value_loss           | 38          |
-----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 38.3     |
|    ep_rew_mean     | 27.4     |
| time/              |          |
|    fps             | 7        |
|    iterations      | 25       |
|    time_elapsed    | 52172    |
|    total_timesteps | 409600   |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 38          |
|    ep_rew_mean          | 27.2        |
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 26          |
|    time_elapsed         | 54249       |
|    total_timesteps      | 425984      |
| train/                  |             |
|    approx_kl            | 0.009729601 |
|    clip_fraction        | 0.121       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.963      |
|    explained_variance   | 0.375       |
|    learning_rate        | 0.0005      |
|    loss                 | 18          |
|    n_updates            | 250         |
|    policy_gradient_loss | 0.00117     |
|    value_loss           | 34.9        |
-----------------------------------------
```

### 11.23
```bash
python3 src/testing.py --env highway --method 0 --epochs 10 --render_mode rgb_array                          
=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 10
Method: OPD (Optimistic Planning)
Random Seed: 42 (deterministic per epoch for fair comparison across agents)
Traffic Density: 1.25
High Speed Reward Weight: 0.4
Collision Reward Weight: -1
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/10: OK, Total reward: 39.68, Avg reward per action: 0.9919
Epoch 2/10: OK, Total reward: 37.53, Avg reward per action: 0.9383
Epoch 3/10: OK, Total reward: 39.95, Avg reward per action: 0.9989
Epoch 4/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 5/10: OK, Total reward: 38.53, Avg reward per action: 0.9633
Epoch 6/10: OK, Total reward: 39.53, Avg reward per action: 0.9883
Epoch 7/10: OK, Total reward: 39.96, Avg reward per action: 0.9989
Epoch 8/10: OK, Total reward: 39.96, Avg reward per action: 0.9989
Epoch 9/10: OK, Total reward: 35.45, Avg reward per action: 0.8861
Epoch 10/10: OK, Total reward: 39.96, Avg reward per action: 0.9991

=== Testing Results ===
Collision Rate (per episode): 0.00%
Collision Rate (per action): 0.0000%
Average Reward (per episode): 39.0506
Average Reward (per action): 0.9763
Average Speed: 29.18 m/s
```

```bash
python3 src/testing.py --env highway --method 0 --epochs 10 --render_mode rgb_array
=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 10
Method: OPD (Optimistic Planning)
Random Seed: 42 (deterministic per epoch for fair comparison across agents)
Traffic Density: 1
High Speed Reward Weight: 0.4
Collision Reward Weight: -1
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 2/10: OK, Total reward: 38.59, Avg reward per action: 0.9648
Epoch 3/10: OK, Total reward: 39.96, Avg reward per action: 0.9991
Epoch 4/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 5/10: OK, Total reward: 39.96, Avg reward per action: 0.9991
Epoch 6/10: OK, Total reward: 37.39, Avg reward per action: 0.9349
Epoch 7/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 8/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 9/10: OK, Total reward: 39.96, Avg reward per action: 0.9990
Epoch 10/10: OK, Total reward: 39.96, Avg reward per action: 0.9990

=== Testing Results ===
Collision Rate (per episode): 0.00%
Collision Rate (per action): 0.0000%
Average Reward (per episode): 39.5674
Average Reward (per action): 0.9892
Average Speed: 29.63 m/s
```




### Unmodified Environment (Default Settings)
```bash
>>> python3 src/testing.py --env highway --method 2 --epochs 20 --duration 40 --render_mode rgb_array

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
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --high_speed_reward_weight 2.0

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
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --collision_reward_weight -5.0

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 100
Method: OPD (Optimistic Planning)
Traffic Density: 1.0
High Speed Reward Weight: 1.0
Collision Reward Weight: -5.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 2/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 3/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 4/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 5/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 6/100: OK, Total reward: 39.76, Avg reward: 0.99
Epoch 7/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 8/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 9/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 10/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 11/100: OK, Total reward: 39.81, Avg reward: 1.00
Epoch 12/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 13/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 14/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 15/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 16/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 17/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 18/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 19/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 20/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 21/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 22/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 23/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 24/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 25/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 26/100: OK, Total reward: 36.23, Avg reward: 0.91
Epoch 27/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 28/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 29/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 30/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 31/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 32/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 33/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 34/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 35/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 36/100: OK, Total reward: 39.43, Avg reward: 0.99
Epoch 37/100: OK, Total reward: 39.39, Avg reward: 0.98
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
Epoch 38/100: CRASHED, Total reward: 22.52, Avg reward: 0.87
Epoch 39/100: OK, Total reward: 39.90, Avg reward: 1.00
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Expanding a terminal state
Epoch 40/100: CRASHED, Total reward: 4.12, Avg reward: 0.82
Epoch 41/100: OK, Total reward: 39.90, Avg reward: 1.00
Epoch 42/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 43/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 44/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 45/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 46/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 47/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 48/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 49/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 50/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 51/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 52/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 53/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 54/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 55/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 56/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 57/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 58/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 59/100: OK, Total reward: 39.56, Avg reward: 0.99
Epoch 60/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 61/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 62/100: OK, Total reward: 39.90, Avg reward: 1.00
Epoch 63/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 64/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 65/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 66/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 67/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 68/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 69/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 70/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 71/100: OK, Total reward: 39.81, Avg reward: 1.00
Epoch 72/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 73/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 74/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 75/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 76/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 77/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 78/100: OK, Total reward: 38.14, Avg reward: 0.95
Epoch 79/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 80/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 81/100: OK, Total reward: 39.81, Avg reward: 1.00
Epoch 82/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 83/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 84/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 85/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 86/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 87/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 88/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 89/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 90/100: OK, Total reward: 39.97, Avg reward: 1.00
Epoch 91/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 92/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 93/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 94/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 95/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 96/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 97/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 98/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 99/100: OK, Total reward: 39.98, Avg reward: 1.00
Epoch 100/100: OK, Total reward: 39.98, Avg reward: 1.00

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 100
Collisions: 2
Collision rate: 2.00%
Average speed: 29.78 m/s
Average total reward: 39.36

=== Summary for Table ===
Collision Rate: 2.00%
Average Speed: 29.78 m/s
Average Reward: 39.36
```

Combined (High Speed + High Penalty):
```bash
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --high_speed_reward_weight 2.0 --collision_reward_weight -5.0
```

### High Traffic Density
```bash
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --traffic_density 1.25
```

### Compare with Fixed Speed Keep Lane Baseline
```bash
python3 src/testing.py --env highway --method 0 --epochs 100 --duration 40 --render_mode rgb_array
```

### OPD Parameter Tuning
Higher budget (more planning)
```bash
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --opd_budget 100
```

Different discount factor
```bash
python3 src/testing.py --env highway --method 2 --epochs 100 --duration 40 --render_mode rgb_array --opd_gamma 0.9
```

# High-speed reward & Collision reward
High-speed reward = 0.25, Collision Reward -2
```bash
python3 src/testing.py --env highway --method 0 --epochs 20 --render_mode rgb_array --collision_reward_weight -2 --high_speed_reward_weigh 0.25

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 20
Method: OPD (Optimistic Planning)
Traffic Density: 1
High Speed Reward Weight: 0.25
Collision Reward Weight: -2.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/20: OK, Total reward: 39.99, Avg reward: 0.9997
Epoch 2/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 3/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 4/20: OK, Total reward: 39.82, Avg reward: 0.9955
Epoch 5/20: OK, Total reward: 38.76, Avg reward: 0.9691
Epoch 6/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 7/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 8/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 9/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 10/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 11/20: OK, Total reward: 39.99, Avg reward: 0.9997
Epoch 12/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 13/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 14/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 15/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 16/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 17/20: OK, Total reward: 39.87, Avg reward: 0.9968
Epoch 18/20: OK, Total reward: 39.98, Avg reward: 0.9996
Epoch 19/20: OK, Total reward: 39.99, Avg reward: 0.9996
Epoch 20/20: OK, Total reward: 39.98, Avg reward: 0.9996

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 20
Collisions: 0
Collision Rate: 0.00%
Average Speed: 29.81 m/s
Average Total Reward (per episode): 39.9100
Average Reward (per step): 0.9978

=== Summary for Table ===
Collision Rate: 0.00%
Average Speed: 29.81 m/s
Average Total Reward: 39.9100
```

High-speed reward = 0.5, Collision Reward -2
```bash
python3 src/testing.py --env highway --method 0 --epochs 10 --render_mode rgb_array --collision_reward_weight -2 --high_speed_reward_weigh 0.5

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 10
Method: OPD (Optimistic Planning)
Traffic Density: 1
High Speed Reward Weight: 0.5
Collision Reward Weight: -2.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/10: OK, Total reward: 39.98, Avg reward: 0.9994
Epoch 2/10: OK, Total reward: 39.97, Avg reward: 0.9993
Epoch 3/10: OK, Total reward: 39.97, Avg reward: 0.9993
Epoch 4/10: OK, Total reward: 39.97, Avg reward: 0.9993
Epoch 5/10: OK, Total reward: 39.97, Avg reward: 0.9993
Epoch 6/10: OK, Total reward: 39.97, Avg reward: 0.9993
Epoch 7/10: OK, Total reward: 39.77, Avg reward: 0.9943
Epoch 8/10: OK, Total reward: 39.97, Avg reward: 0.9994
Epoch 9/10: OK, Total reward: 39.97, Avg reward: 0.9994
Epoch 10/10: OK, Total reward: 39.52, Avg reward: 0.9879

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 10
Collisions: 0
Collision Rate: 0.00%
Average Speed: 29.89 m/s
Average Total Reward (per episode): 39.9076
Average Reward (per step): 0.9977

=== Summary for Table ===
Collision Rate: 0.00%
Average Speed: 29.89 m/s
Average Total Reward: 39.9076
Average Per-Step Reward: 0.9977
```

High-speed reward = 0.75, Collision Reward -2
```bash
python3 src/testing.py --env highway --method 0 --epochs 10 --render_mode rgb_array --collision_reward_weight -2 --high_speed_reward_weigh 0.75

=== Testing Settings ===
Environment: highway
Duration: 40s
Epochs: 10
Method: OPD (Optimistic Planning)
Traffic Density: 1
High Speed Reward Weight: 0.75
Collision Reward Weight: -2.0
OPD Budget: 50
OPD Gamma: 0.7

OPD Agent initialized with budget=50, gamma=0.7

Epoch 1/10: OK, Total reward: 39.96, Avg reward: 0.9990
Epoch 2/10: OK, Total reward: 39.97, Avg reward: 0.9991
Epoch 3/10: OK, Total reward: 39.96, Avg reward: 0.9991
Epoch 4/10: OK, Total reward: 39.96, Avg reward: 0.9990
Epoch 5/10: OK, Total reward: 39.96, Avg reward: 0.9991
Epoch 6/10: OK, Total reward: 39.96, Avg reward: 0.9990
Epoch 7/10: OK, Total reward: 39.96, Avg reward: 0.9990
Epoch 8/10: OK, Total reward: 39.96, Avg reward: 0.9990
Epoch 9/10: OK, Total reward: 39.96, Avg reward: 0.9991
Epoch 10/10: OK, Total reward: 39.96, Avg reward: 0.9991

=== Testing Results ===
Method: OPD (Optimistic Planning)
Total epochs: 10
Collisions: 0
Collision Rate: 0.00%
Average Speed: 29.97 m/s
Average Total Reward (per episode): 39.9620
Average Reward (per step): 0.9991

=== Summary for Table ===
Collision Rate: 0.00%
Average Speed: 29.97 m/s
Average Total Reward: 39.9620
Average Per-Step Reward: 0.9991
```