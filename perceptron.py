import numpy as np

class Perceptron:
    weights = None
    threshold = 0
    nNeurons = 0

    def __init__(self, nNeurons, nInputs, initial_weight = 1):
        self.weights = np.full((nInputs+1, nNeurons), initial_weight)
        self.nNeurons = nNeurons

    def train(self, inputs, targets, learning_rate = 1, epochs=1, print_epochs=True):
        # inputs_and_bias
        inputs_bias = np.concatenate((inputs,np.ones((inputs.shape[0], 1))), axis=1)
        for epoch in range(epochs):
            outputs = self.test(inputs_bias, False)
            delta_W = - learning_rate * (inputs_bias.T @ (outputs-targets))
            self.weights += delta_W
            if print_epochs:
                print(f'Epoch {epoch}')
                print(self.weights)
                print('Final outputs')
                print(outputs)

    def test(self, inputs, add_bias=True):
        if add_bias:
            inputs_bias = np.concatenate((inputs,np.ones((inputs.shape[0], 1))), axis=1)
        else:
            inputs_bias = inputs
        nets = (inputs_bias @ self.weights)
        return np.where(nets > 0, 1, 0)
