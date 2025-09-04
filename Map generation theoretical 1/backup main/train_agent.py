# train_agent.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# --------------------
# Q-Network (CNN for 2D map)
# --------------------
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim=128):
        """
        input_shape: (H, W)
        output_dim: number of actions
        """
        super(QNetwork, self).__init__()
        C, H, W = 1, input_shape[0], input_shape[1]  # single-channel
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 64 * H * W
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, H, W) -> add channel
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dim
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --------------------
# Replay Buffer (store 2D grids)
# --------------------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        if next_state is not None:
            next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        next_states = np.array([ns if ns is not None else np.zeros_like(states[0], dtype=np.float32) 
                                for ns in next_states], dtype=np.float32)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            next_states,
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------
# RL Agent
# --------------------
class RLAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, lr=1e-3, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000):

        self.state_size = state_size  # tuple (H, W)
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_target()

        # Replay memory
        self.memory = ReplayBuffer()

        # Epsilon-greedy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0

    def current_epsilon(self):
        fraction = min(self.total_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def act(self, state, epsilon):
        self.total_steps += 1
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        if done:
            next_state = np.zeros_like(state, dtype=np.float32)
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()
