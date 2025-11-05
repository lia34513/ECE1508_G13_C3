# ECE1508_G13_C3_RL for Autonomous Driving
Advancements in reinforcement learning (RL) and deep neural networks (DNNs) are tackling real-world challenges like robotic control, traffic signal management, and portfolio optimization. Autonomous driving is a standout application, demanding decision-making under uncertainty to balance safety, efficiency, and comfort.

We design and implement an RL agent for basic autonomous driving tasks such as lane-keeping, overtaking, or collision avoidance. The project aims to compare RL-based driving policies with simple rule-based baselines in lightweight simulators such as [Highway-env](https://github.com/Farama-Foundation/HighwayEnv/tree/master).

We compare the following factors to perform ablation experiments:

* Rule-based baselines vs. RL agents
* Different scenarios: Highway, Merge, Roundabout, and Parking
* Introduce modifications: Gaussian noise in sensor observations and delays/perturbations in actions

We evaluate and analyze driving performance metrics, including collisions and average speed, while comparing RL agents against baselines under different conditions.

