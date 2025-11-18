"""
Simple DQN agent for discrete-action environments.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Small fully-connected network for approximating Q-values."""

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Minimal DQN agent with:
    - epsilon-greedy policy
    - replay buffer
    - target network
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        memory_capacity=50000,
        batch_size=64,
        target_update_interval=10,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        # Online and target networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Simple replay buffer implemented with NumPy arrays
        self.memory_capacity = memory_capacity
        self.memory_counter = 0
        self.states = np.zeros((memory_capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((memory_capacity,), dtype=np.int64)
        self.rewards = np.zeros((memory_capacity,), dtype=np.float32)
        self.next_states = np.zeros((memory_capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((memory_capacity,), dtype=np.float32)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state: 1D NumPy array representing the current state.

        Returns:
            Integer action index.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store one transition in the replay buffer."""
        idx = self.memory_counter % self.memory_capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.memory_counter += 1

    def can_provide_sample(self):
        """Return True if the buffer has enough samples for one batch."""
        return self.memory_counter >= self.batch_size

    def sample_batch(self):
        """Sample one batch from the replay buffer."""
        max_mem = min(self.memory_counter, self.memory_capacity)
        indices = np.random.choice(max_mem, self.batch_size, replace=False)

        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices]).long()
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).float()

        return states, actions, rewards, next_states, dones

    def train_step(self):
        """
        Perform one gradient descent step using a sampled mini-batch.

        Returns:
            Loss value as a float, or None if not enough samples yet.
        """
        if not self.can_provide_sample():
            return None

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Q(s, a) from online network
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def load_dqn_agent_from_checkpoint(state_dim, action_dim, checkpoint_path):
    """
    Load a trained DQN agent from a checkpoint file.

    Args:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        DQNAgent: Loaded agent with trained weights, or None if loading fails.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Create agent with same hyperparameters as training
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        memory_capacity=50000,
        batch_size=64,
        target_update_interval=10,
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return agent
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}") from e