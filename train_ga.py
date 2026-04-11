import numpy as np
import sys
from flappy_genetic_algo import run_genetic_algo
from flappy_neural_network import FlappyNeuralNetwork
from flappy_bird_env import FlappyBirdEnv
import matplotlib.pyplot as plt


def train(layers, population_size, generations, mutation_rate):
    print("Training GA...")
    best_genome, best_history, avg_history = run_genetic_algo(layers, population_size, generations, mutation_rate)

    # Save the best genome
    np.save("best_genome.npy", best_genome)
    print(best_genome)
    print(best_history)
    print(avg_history)

    print("Best genome saved to best_genome.npy")

    # Plot results
    plt.plot(best_history, label="Best Fitness")
    plt.plot(avg_history, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Performance")
    plt.legend()
    title = f"genetic algo plots/ga_performance pop_{population_size}_gen_{generations}_mut_{mutation_rate}.png" 
    plt.savefig(title, dpi=300)
    plt.close() # Prevents blocking popup window

def render(layers):
    print("Rendering best genome...")
    best_genome = np.load("best_genome.npy")
    net = FlappyNeuralNetwork(layers)
    net.set_weights(best_genome)

    env = FlappyBirdEnv(render_mode="human")
    obs, _ = env.reset()

    while True:
        action = net.forward_propagation(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Final Score: {info['score']}")
            break

    env.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    # hyperparameters
    layers = [5, 12, 2]
    population_size = 350
    generations = 50
    mutation_rate = 0.1

    if args.train:
        train(layers, population_size, generations, mutation_rate)
    else:
        render(layers)