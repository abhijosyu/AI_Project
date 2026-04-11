import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flappy_bird_env import FlappyBirdEnv


def build_q_network(state_dim, action_dim, hidden_size=256):
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_dim)
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

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    q_values = q_network(state_tensor).detach()

    return torch.argmax(q_values, dim=1).item()

def train_step(batch, q_net, target_net, optimizer, loss_fn, gamma, device, max_norm=10):
    states, actions, rewards, next_states, dones = batch

    # Convert to tensors
    states = torch.from_numpy(states).to(device)
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.from_numpy(rewards).to(device)
    next_states = torch.from_numpy(next_states).to(device)
    dones = torch.from_numpy(dones).to(device)

    # Current Q-values from network
    current_q = q_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

    # Get target Q-values from frozen target network
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target_q = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()

    # Log gradient norm before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=max_norm)

    optimizer.step()

    return loss.item(), grad_norm.item()

def sync_target_network(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())

def epsilon_decay(epsilon, epsilon_min, epsilon_decay_rate):
    return max(epsilon_min, epsilon * epsilon_decay_rate)