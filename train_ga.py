import numpy as np
import sys
from flappy_genetic_algo import run_genetic_algo
from flappy_neural_network import FlappyNeuralNetwork
from flappy_bird_env import FlappyBirdEnv

train_flag = 'train' in sys.argv

# hyperparameters
layers = [5, 8, 2]
population_size = 200
generations = 50
mutation_rate = 0.1

# Train
if train_flag:

    print("Training GA...")
    best_genome = run_genetic_algo(layers, population_size, generations, mutation_rate)

    # Save the best genome
    np.save("best_genome.npy", best_genome)
    print("Best genome saved to best_genome.npy")

# Watch it play
else: 
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