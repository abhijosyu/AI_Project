from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from flappy_bird_env import FlappyBirdEnv
from flappy_dqn import (
    build_q_network,
    add_to_replay_buffer,
    sample_from_replay_buffer,
    select_action,
    train_step,
    sync_target_network,
    epsilon_decay
)

train_flag = 'train' in sys.argv

# Hyperparameters
BUFFER_SIZE        = 50000
BATCH_SIZE         = 64
GAMMA              = 0.99
EPSILON_START      = 1.0
EPSILON_MIN        = 0.01
EPSILON_DECAY_RATE = 0.9999  # per-step decay (slower, more consistent)
TARGET_UPDATE_FREQ = 200
WARMUP_STEPS       = 1000
MAX_EPISODES       = 1000
LEARNING_RATE      = 1e-4
PATIENCE           = 500     # early stopping if no improvement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if train_flag:
    print("Training DQN...")
    env = FlappyBirdEnv()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = build_q_network(state_dim, action_dim).to(device)
    target_net = build_q_network(state_dim, action_dim).to(device)
    sync_target_network(q_net, target_net)

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    epsilon = EPSILON_START
    total_steps = 0
    best_reward = -float('inf')
    episodes_since_improvement = 0

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        loss = None
        grad_norm = None

        while True:
            action = select_action(state, q_net, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            add_to_replay_buffer(replay_buffer, state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Decay epsilon per step for consistent exploration control
            epsilon = epsilon_decay(epsilon, EPSILON_MIN, EPSILON_DECAY_RATE)

            if len(replay_buffer) >= BATCH_SIZE and total_steps >= WARMUP_STEPS:
                batch = sample_from_replay_buffer(replay_buffer, BATCH_SIZE)
                loss, grad_norm = train_step(batch, q_net, target_net, optimizer, loss_fn, GAMMA, device)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                sync_target_network(q_net, target_net)

            if done:
                break

        # Logging
        grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | Epsilon: {epsilon:.3f} | Loss: {loss_str} | Grad Norm: {grad_str}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            episodes_since_improvement = 0
            torch.save(q_net.state_dict(), "best_dqn.pth")
            print(f"  -> New best reward: {best_reward:.2f}, model saved!")
        else:
            episodes_since_improvement += 1

        # Early stopping
        if episodes_since_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} episodes. Stopping early.")
            break

    print(f"Training complete. Best reward: {best_reward:.2f}")

else:
    print("Rendering DQN agent...")
    env = FlappyBirdEnv(render_mode="human")
    state, _ = env.reset()
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = build_q_network(state_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load("best_dqn.pth", map_location=device))
    q_net.eval()

    while True:
        action = select_action(state, q_net, epsilon=0.0, action_dim=action_dim, device=device)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Final Score: {info['score']}")
            break

    env.close()