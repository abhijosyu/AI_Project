# CS4100 Final Project — Flappy Bird AI

**Team:** Abhiram Josyula, Jessica Jain, Jaylen Zeng

<![App Demo](assets/flappy_demo_V1.gif)>

## Overview

This project explores training an AI agent to play Flappy Bird using two different machine learning approaches. We built a custom Flappy Bird environment from scratch using the Gymnasium API, and then trained agents using both Deep Q-Learning and a Genetic Algorithm. The goal was to compare how each method learns to navigate the game over time.

## How to Run

Before attempting to run this project, make sure you have Python 3.14 installed.

``` python --version ```

If not installed, install Python 3.14 depending on your OS.

If you have Python installed, download all dependencies using: 

``` pip install -r requirements.txt```

Run each agent using the following commands:

For Reinforcement Learning:

``` 
python train_dqn.py train // for training
python train_dqn.py // to visualize once trained
```

For  Genetic Algorithms:

```
python train_ga.py --train
python train_ga.py
```

## Approaches

**Deep Q-Network (DQN)**
- Uses reinforcement learning to train a neural network agent
- Learns by playing the game repeatedly and updating based on rewards
- Key techniques: experience replay buffer, epsilon-greedy exploration, target network
- Tuned hyperparameters including learning rate, buffer size, batch size, and discount factor

**Genetic Algorithm (GA)**
- Evolves a population of neural networks across many generations
- No gradient descent — the best-performing birds pass their weights forward
- Tested various population sizes and mutation rates to study convergence behavior
- Generates performance plots comparing best vs. average fitness per generation

## Files

- `flappy_bird_env.py` — Custom Flappy Bird game environment
- `train_dqn.py` — Train or run the DQN agent
- `train_ga.py` — Train or run the Genetic Algorithm agent
- `flappy_dqn.py` — DQN network and training utilities
- `flappy_genetic_algo.py` — GA selection, crossover, and mutation logic
- `flappy_neural_network.py` — Neural network used by the GA agent
- `rl plots/` — DQN training performance plots
- `genetic algo plots/` — GA experiment results
- `tests/` — Unit tests for the environment and model
