import numpy as np

layers = [5, 8, 2]

class FlappyNeuralNetwork():
    
    def __init__(self, layers, activation="tanh"):
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])
            b = np.zeros(layers[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward_propagation(self, obs):
        activations = obs # Initial input data
        
        # Loop through each layer in the network
        for i in range(len(self.weights)):
            weighted_sum = np.dot(activations, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                activations = np.tanh(weighted_sum)
            else:
                activations = weighted_sum
        return np.argmax(activations)
    
    # Our Genetic Algorithm needs a 1D array as the genome so we need to flatten the matrices.
    def get_weights(self):
        parts = []
        for i in range(len(self.weights)):
            parts.append(self.weights[i].flatten())
            parts.append(self.biases[i].flatten())
        return np.concatenate(parts)
    
    def set_weights(self, genome):
        idx = 0
        for i in range(len(self.layers) - 1):
            w_size = self.layers[i] * self.layers[i + 1]
            b_size = self.layers[i + 1]

            w = genome[idx:idx + w_size].reshape(self.layers[i], self.layers[i + 1])
            idx += w_size

            b = genome[idx:idx + b_size]
            idx += b_size

            self.weights[i] = w
            self.biases[i] = b