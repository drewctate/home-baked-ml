import numpy as np
import math

class Helpers:
    def sigmoid(n):
        return 1.0 / (1.0 + math.exp(-n))

    def f_prime(output):
        return output * (1.0 - output)

    def random_numbers_close_to_zed(n_numbers):
        """Returns random weights close to zero"""
        min_val, max_val = -.1, .1
        return (max_val - min_val) * np.random.random_sample((n_numbers,)) + min_val


class NeuralNetwork:
    learning_rate = None
    num_inputs = None
    input_bias = None
    hidden_layer = None
    output_layer = None

    def __init__(self, num_inputs, num_hidden, num_output, learning_rate=.5, init_hidden_weights=None, init_output_weights=None, input_bias=1.0, hidden_bias=1.0):
        self.learning_rate = learning_rate
        self.input_bias = input_bias
        self.num_inputs = num_inputs
        self.hidden_layer = self.init_neuron_layer(num_hidden, num_inputs, init_hidden_weights, hidden_bias)
        self.output_layer = self.init_neuron_layer(num_output, num_hidden, init_output_weights, None)

    def train(self, inputs, targets, print_steps=False):
        inputs_w_bias = np.concatenate((inputs, np.array([self.input_bias])))
        outputs = self.forward_propagate(inputs_w_bias)
        if print_steps: print(f'Outputs: {outputs}')

        # Output layer deltas
        out_deltas = []
        for i, neuron in enumerate(self.output_layer.neurons):
            out_deltas.append((targets[i] - neuron.out) * Helpers.f_prime(neuron.out))

        if print_steps: print(f'Out Layer Deltas: {out_deltas}')

        # Hidden layer deltas
        hidden_deltas = []
        for j in range(len(self.hidden_layer.neurons) + 1):
            out_j = None
            if j < len(self.hidden_layer.neurons):          # If this is one of the actual neurons
                out_j = self.hidden_layer.neurons[j].out
            else:                                           # Else it is the bias
                out_j = self.hidden_layer.bias

            sum_of_child_deltas_times_weights = 0

            for k in range(len(self.output_layer.neurons)):
                sum_of_child_deltas_times_weights += out_deltas[k] * self.output_layer.neurons[k].weights[j]

            hidden_deltas.append(sum_of_child_deltas_times_weights * Helpers.f_prime(out_j))

        if print_steps: print(f'Hidden Layer Deltas: {hidden_deltas}')

        # Update hidden-out weights
        for i in range(len(self.hidden_layer.neurons) + 1):
            if i < len(self.hidden_layer.neurons):          # If this is one of the actual neurons
                out_i = self.hidden_layer.neurons[i].out
            else:                                           # Else it is the bias
                out_i = self.hidden_layer.bias

            for j, neuron in enumerate(self.output_layer.neurons):
                delta_w = self.learning_rate * out_i * out_deltas[j]
                if print_steps: print(f'hidden-out {i}-{j}: {delta_w}')
                neuron.weights[i] += delta_w

        # Update input-hidden weights
        for i in range(len(inputs_w_bias)):
            out_i = inputs_w_bias[i]

            for j, neuron in enumerate(self.hidden_layer.neurons):
                delta_w = self.learning_rate * out_i * hidden_deltas[j]
                if print_steps: print(f'input-hidden {i}-{j}: {delta_w}')
                neuron.weights[i] += delta_w

    def predict(self, inputs):
        inputs_w_bias = np.concatenate((inputs, np.array([self.input_bias])))
        return self.forward_propagate(inputs_w_bias)

    def inspect(self):
        print('------')
        print(f'Inputs: {self.num_inputs}')
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('Output Layer')
        self.output_layer.inspect()
        print('------')

    # Expects inputs including bias
    def forward_propagate(self, inputs):
        hidden_layer_out = self.hidden_layer.get_outputs_w_bias(inputs)
        output_layer_out = self.output_layer.get_outputs(hidden_layer_out)
        return output_layer_out

    def init_neuron_layer(self, num_neurons, num_weights, init_weights, bias):
        return NeuronLayer(num_neurons, num_weights, init_weights, bias)

    def f_prime(self, output):
        return


class NeuronLayer:
    neurons = None
    bias = None

    def __init__(self, num_neurons, num_weights, init_weights, bias = None):
        self.bias = bias

        if init_weights is None:
            init_weights = Helpers.random_numbers_close_to_zed(num_weights + 1) # (Add an extra weight for the bias)

        self.neurons = [Neuron(np.copy(init_weights)) for i in range(num_neurons)]

    def get_outputs(self, inputs):
        return np.array([neuron.get_output(inputs) for neuron in self.neurons])

    def get_outputs_w_bias(self, inputs):
        return np.concatenate((self.get_outputs(inputs), np.array([self.bias])))

    def inspect(self):
        print('------')
        print(f'Number of Neurons: {len(self.neurons)}')
        print('------')
        print('Neurons')
        for i, neuron in enumerate(self.neurons):
            print(f'Neuron {i}')
            print(f'Weights: {neuron.weights}')


class Neuron:
    weights = None
    inputs = None
    net = None
    out = None

    def __init__(self, weights):
        self.weights = weights

    def get_output(self, inputs):
        self.inputs = inputs
        self.net = np.sum(inputs * self.weights)
        self.out = self.get_activation(self.net)
        return self.out

    def get_activation(self, net):
        return Helpers.sigmoid(net)

