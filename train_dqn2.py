import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flappy_bird_env import FlappyBirdEnv


def build_q_network(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim)
    )


def add_to_replay_buffer(replay_buffer, state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))


def sample_from_replay_buffer(replay_buffer, batch_size):
    batch = random.sample(replay_buffer, batch_size)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for experience in batch:
        state, action, reward, next_state, done = experience

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32)
    )


def select_action(state, q_network, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

		q_values = q_network(state_tensor).detach()

    return torch.argmax(q_values, dim=1).item()