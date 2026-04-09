import numpy as np
import random

from flappy_bird_env import FlappyBirdEnv
from flappy_neural_network import FlappyNeuralNetwork

def evaluate_fitness(genome, layers):
    max_frames = 3000
    frames = 0
    
    net = FlappyNeuralNetwork(layers)
    net.set_weights(genome)
    
    env = FlappyBirdEnv()
    obs, _ = env.reset()
    total_reward = 0
    
    while frames < max_frames:
        action = net.forward_propagation(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames += 1
        if terminated:
            break
    
    env.close()
    return total_reward

def run_genetic_algo(layers, population_size, generations, mutation_rate):
    # Create initial populate of random genomes
    genome_length = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
    population = [np.random.randn(genome_length) for _ in range(population_size)]

    best_history = []
    avg_history = []

    
    for gen in range(generations):
        # Evaluate fitness for each genome
        fitness_scores = [evaluate_fitness(g, layers) for g in population]
        print(f"Gen {gen} | Best: {max(fitness_scores):.2f} | Avg: {np.mean(fitness_scores):.2f}")
        
        best = max(fitness_scores)
        avg = np.mean(fitness_scores)

        best_history.append(best)
        avg_history.append(avg)

        # Select the best ones (keeps top 20%)
        top_k = population_size // 5
        ranked = np.argsort(fitness_scores)[::-1]
        parents = [population[i] for i in ranked[:top_k]]
                
        # Form new population
        new_population = list(parents)
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            # Crossover to create new children
            child = crossover(p1, p2)
            # Mutate children
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    
    fitness_scores = [evaluate_fitness(g, layers) for g in population]
    best_idx = np.argmax(fitness_scores)
    return population[best_idx], best_history, avg_history
        
def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child

def mutate(genome, mutation_rate):
    for i in range(len(genome)):
        if np.random.random() < mutation_rate:
            genome[i] += np.random.randn() * 0.5
    return genome

