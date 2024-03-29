import numpy as np 
from mlxtend.data import loadlocal_mnist

class model():
    def __init__(self):
        self.z_vectors = []
        self.activation_vectors = []
        self.training_rate = .1
        # number of nodes by layer 
        self.node_count = [784, 150, 150, 10]
        self.weights = []
        self.bias = []
        for i in range(len(self.node_count) - 1):
            self.weights.append(np.random.rand(self.node_count[i + 1], self.node_count[i]))
            self.bias.append(np.random.rand(self.node_count[i + 1], 1))
        print(self.bias[0].size)
        print(self.bias[1].size)
        self.images, self.labels = loadlocal_mnist(
        images_path='data/train-images.idx3-ubyte',
        labels_path='data/train-labels.idx1-ubyte'
        )
        

    def recieve_data(self, data):
        entry = self.color.tolist() + data
        self.data.append(entry)
        for i in range(1000):
            self.train()
        self.color = np.random.choice(range(256), size=3)
        self.make_guess()
        return self.color, self.confidence, self.guess

    def make_guess(self):
        predictions = self.NN(self.color).flatten()
        if predictions[0] > predictions[1]:
            self.guess = 1
        else: 
            self.guess = 0
        self.confidence = 100 - 50 * ((predictions[0] - self.guess) ** 2 + (predictions[1] - (1 - self.guess)) ** 2)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def dsig(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
        
    def NN(self, value):
        z = 1/255 * value.reshape(-1, 1)
        self.activation_vectors, self.z_vectors = [], []
        self.activation_vectors.append(np.asarray(z))
        for i in range(len(self.node_count) - 1):
            self.z_vectors.append(np.asarray([(1/len(self.activation_vectors[i])) * (self.weights[i] @ self.activation_vectors[i]) + self.bias[i]]).reshape(-1, 1))
            self.activation_vectors.append(self.sigmoid(self.z_vectors[i]).reshape(-1, 1))
        return self.activation_vectors[-1].reshape(-1, 1)

    def train(self):
        idx = np.random.choice(len(self.images))
        value = np.asarray(self.images[idx]).reshape(-1, 1)
        target = np.zeros((10, 1))
        target[self.labels[idx] - 1][0] = 1
        self.error_signal_vectors = []
        
        predictions = self.NN(value)

        error_signal = []
        # Get error signals of last layer 
        for i in range(len(predictions)):
            dcost = 2 * (predictions[i] - target[i])
            # Get error signal of specfic node
            err_sig = dcost * self.dsig(self.z_vectors[-1][i])
            # Add to list for this layer
            error_signal.append(err_sig)
        
        self.error_signal_vectors.append(np.asarray(error_signal).reshape(-1, 1))
        
        # Find error signals for previous layers
        for i in range(len(self.node_count) - 2):
            next_err_v = self.weights[-(i + 1)].transpose() @ self.error_signal_vectors[i]
            self.error_signal_vectors.append(next_err_v)
        
        # Build change in weight matrices
        dweights = []
        print(self.error_signal_vectors[2])
        for i in range(len(self.node_count) - 1):
            matrix = self.error_signal_vectors[i] @ self.activation_vectors[-(i + 2)].reshape(1, -1)
            dweights.append(np.asarray(matrix))
        print(np.amax(dweights[2]))
        

        # Update weights and biases
        for i in range(len(self.node_count) - 1):
            self.weights[-(i + 1)] -= self.training_rate * dweights[i]
            self.bias[-(i + 1)] -= self.training_rate * self.error_signal_vectors[i]
            
        return predictions, target

class view():
    def __init__():
        self.canvas = []

class controller():
    def __init__():
        self.v = []

if __name__ == '__main__':
    model = model()
    model.train()