"""
Rule-based baseline policies for autonomous driving.
"""

from rl_agents.agents.common.factory import agent_factory


def fixed_speed_keep_lane(env):
    """Fixed speed keep lane action - maintains current lane and speed (IDLE)."""
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    return action


class OPDAgent:
    """
    Optimistic Planning for Deterministic systems (OPD) agent.

    This agent uses tree search with optimistic planning to find optimal actions
    in deterministic environments like Highway-env.

    Reference: https://hal.science/hal-00830182/document
    """

    def __init__(self, env, budget=50, gamma=0.7):
        """
        Initialize OPD agent.

        Args:
            env: The gymnasium environment
            budget: Number of planning expansions (default: 50)
            gamma: Discount factor for future rewards (default: 0.7)
        """
        self.env = env
        self.budget = budget
        self.gamma = gamma

        # Configure the OPD agent using rl-agents library
        agent_config = {
            "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
            "env_preprocessors": [{"method": "simplify"}],
            "budget": self.budget,
            "gamma": self.gamma,
        }

        self.agent = agent_factory(env, agent_config)

    def act(self, observation):
        """
        Get the optimal action for the current observation.

        Args:
            observation: Current environment observation

        Returns:
            action: The optimal action determined by OPD planning
        """
        return self.agent.act(observation)

    def reset(self):
        """Reset the agent's internal state."""
        if hasattr(self.agent, 'reset'):
            self.agent.reset()
