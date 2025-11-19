import time
import gymnasium
import highway_env
from env_config import get_highway_config

"""
Test the execution speed (steps per second) of the highway-env environment.

This script performs the following:
1. Loads the highway-v0 environment using the project’s configuration.
2. Executes a fixed number of environment steps (default: 200) using random actions.
3. Measures the total time taken to complete these steps.
4. Prints intermediate progress every 50 steps.
5. Reports the total elapsed time and the approximate FPS (environment steps per second).

This utility is used to benchmark environment performance and diagnose whether
environment computation speed is a bottleneck for RL training.
"""

config = get_highway_config()
env = gymnasium.make("highway-v0", config=config)
obs, info = env.reset()

n = 200  # test 200 steps
t0 = time.time()
for i in range(n):
    action = env.action_space.sample()
    obs, r, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
    if (i + 1) % 50 == 0:
        print(f"progress: {i + 1}/{n}")
t1 = time.time()

print(f"{n} steps in {t1 - t0:.2f} seconds")
print(f"fps ≈ {n / (t1 - t0):.1f}")