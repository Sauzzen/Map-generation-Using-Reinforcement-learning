# dqn_sprite_agent_large_map.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from sprite_tilemap_env import SpriteTilemapEnv, LAND, WATER
import pygame

# ------------------------------
# Hyperparameters
# ------------------------------
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 5000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
INITIAL_LAND_ONLY_EPISODES = 5  # train on LAND/WATER first

# ------------------------------
# Simple Feedforward DQN
# ------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ------------------------------
# Replay Memory
# ------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ------------------------------
# DQN Agent
# ------------------------------
class DQNAgent:
    def __init__(self, env: SpriteTilemapEnv, land_only=True, batch_size=32):
        self.env = env
        self.view_width = env.viewport_width
        self.view_height = env.viewport_height
        self.all_tile_types = list(env.sprites.keys())
        self.tile_types = [LAND, WATER] if land_only else self.all_tile_types
        self.num_tile_types = len(self.tile_types)

        # State = viewport flattened
        self.state_size = self.view_width * self.view_height
        # Action = map_width * map_height * num_tile_types
        self.action_size = env.map_width * env.map_height * self.num_tile_types

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.batch_size = batch_size
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.criterion = nn.MSELoss()

    # --------------------------
    def state_to_tensor(self, state):
        return torch.tensor(state.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)

    # --------------------------
    def choose_action(self, state):
        if random.random() < self.epsilon:
            x = random.randrange(self.env.map_width)
            y = random.randrange(self.env.map_height)
            tile_type = random.choice(self.tile_types)
            return x, y, tile_type

        with torch.no_grad():
            q_values = self.policy_net(self.state_to_tensor(state)).squeeze(0)

            # Mask invalid actions
            mask = torch.full_like(q_values, float('-inf'))
            allowed_indices = [
                self.action_to_idx(x, y, t)
                for x in range(self.env.map_width)
                for y in range(self.env.map_height)
                for t in self.tile_types
            ]
            mask[allowed_indices] = 0
            idx = (q_values + mask).argmax().item()

        return self.idx_to_action(idx)

    # --------------------------
    def idx_to_action(self, idx):
        tile_idx = idx % self.num_tile_types
        pos = idx // self.num_tile_types
        y = pos // self.env.map_width
        x = pos % self.env.map_width
        return (x, y, self.tile_types[tile_idx])

    # --------------------------
    def action_to_idx(self, x, y, tile_type):
        tile_idx = self.tile_types.index(tile_type)
        pos = y * self.env.map_width + x
        return pos * self.num_tile_types + tile_idx

    # --------------------------
    def store_transition(self, state, action, reward, next_state, done):
        idx = self.action_to_idx(*action)
        # 🔑 clip to avoid out of bounds indexing
        idx = min(idx, self.action_size - 1)
        self.memory.push(state, idx, reward, next_state, done)

    # --------------------------
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory.sample(self.batch_size))

        states = torch.stack([torch.tensor(s.flatten(), dtype=torch.float32) for s in states]).to(self.device)
        next_states = torch.stack([torch.tensor(ns.flatten(), dtype=torch.float32) for ns in next_states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).clamp(0, self.action_size - 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --------------------------
    def enable_all_tiles(self):
        self.tile_types = self.all_tile_types
        self.num_tile_types = len(self.tile_types)
        self.action_size = self.env.map_width * self.env.map_height * self.num_tile_types
        # 🔑 reset networks with correct action_size
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

# ------------------------------
def train_dqn(episodes=20):
    env = SpriteTilemapEnv(map_width=20, map_height=20, viewport_width=8, viewport_height=8,
                           sprite_size=50, debug=False)
    agent = DQNAgent(env, land_only=True)

    for ep in range(episodes):
        env.reset(biome=LAND if ep < INITIAL_LAND_ONLY_EPISODES else None)
        done, step = False, 0

        if ep == INITIAL_LAND_ONLY_EPISODES:
            agent.enable_all_tiles()

        while not done:
            state = env.tilemap[env.offset_y:env.offset_y+env.viewport_height,
                                 env.offset_x:env.offset_x+env.viewport_width]

            x, y, tile_type = agent.choose_action(state)
            env.place_sprite(x, y, tile_type)

            # camera auto-scroll
            if x < env.offset_x: env.scroll(x - env.offset_x, 0)
            elif x >= env.offset_x + env.viewport_width: env.scroll(x - (env.offset_x + env.viewport_width - 1), 0)
            if y < env.offset_y: env.scroll(0, y - env.offset_y)
            elif y >= env.offset_y + env.viewport_height: env.scroll(0, y - (env.offset_y + env.viewport_height - 1))

            next_state = env.tilemap[env.offset_y:env.offset_y+env.viewport_height,
                                     env.offset_x:env.offset_x+env.viewport_width]

            reward = env.compute_reward(x, y, tile_type)
            done = step >= env.map_width * env.map_height - 1

            agent.store_transition(state, (x, y, tile_type), reward, next_state, done)
            agent.train_step()
            step += 1

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        if ep % 5 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode {ep+1}/{episodes} finished | Epsilon={agent.epsilon:.3f}")

    # free-roam camera after training
    running = True
    while running:
        env.render()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: env.scroll(-1, 0)
        if keys[pygame.K_d]: env.scroll(1, 0)
        if keys[pygame.K_w]: env.scroll(0, -1)
        if keys[pygame.K_s]: env.scroll(0, 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        env.clock.tick(10)
    env.close()
    return agent

# ------------------------------
if __name__ == "__main__":
    trained_agent = train_dqn(episodes=400)
2