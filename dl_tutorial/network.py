"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

It is used to calculate param in the neural networks.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np
from numpy import linalg as LA

import joblib
import mnist_loader

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, 
          [2, 3, 1] --- 3-layers with 2/3/1 neurons respectively.
          
        weights and biases are params between layers.
        X ---> sigmoid(w dot X + b)  ---> a
        For the example of [2, 3, 1]:

        formula between layer 1 and layer 2:
        [x1, x2] 
             sigmoid([w11, w12] dot [x1, x2] + b1) ---> a1
        ---> sigmoid([w21, w22] dot [x1, x2] + b2) ---> a2
             sigmoid([w31, w32] dot [x1, x2] + b3) ---> a3
        params between layer 1 and layer 2:
          w = [[w11, w12],          b = [b1,
               [w21, w22],               b2,
               [w31, w32]]               b3]

        formula between layer 2 and layer 3:
        [a1, a2, a3] 
         --> sigmoid([w1, w2, w3] dot [a1, a2, a3] + b) ---> y
        params between layer 2 and layer 3:
          w = [w1, w2, w3]    b = b
        """
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.costs = [] # used for investigating cost performance after some steps
        self.check_cost_inside_SGD = False
    
    def fit(self, X):
        return self.SGD(X)
    
    def predict(self, a):
        return self.feedforward(a)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs=30, mini_batch_size=10, eta=3.0,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        start = time.time()

        # cost data are used only for investigation
        self.costs = []
        cost = self.average_cost(training_data)
        self.costs.append((0, cost))
        print("Epoch {0}: initial average cost {1}".format(0, cost))

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for idx, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(mini_batch, eta)
                # collect cost value after consuming some inputs
                count = (idx+1) * mini_batch_size
                if ( j == 0 and self.check_cost_inside_SGD and
                     (idx < 50 or 
                      (count <= 6000 and count % 100 == 0) or 
                      count % 1000 == 0)
                   ):
                    cost = self.average_cost(training_data)
                    self.costs.append((count,cost))
                    print("Epoch {0} count {1}: updated average cost {2}".format(
                          j, count , cost))
                
            # we can stop the training if cost is small enough
            # but it is used only for investigation currently.
            cost = self.average_cost(training_data)
            self.costs.append((n*(j+1),cost))
            print("Epoch {0}: updated average cost {1}".format( j, cost))
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
        end = time.time()
        return (end-start)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # sigmoid derivative can reuse the activation in feedforward 
        # sigmoid(z)' = sigmoid(z) * (1 - sigmoid(z))
        sd = activation * (1 - activation)
        # for the output layer
        delta = self.cost_derivative(activations[-1], y) * sd
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # for the hidden layers
        for l in range(2, self.num_layers):
            sd = activations[-l] * (1 - activations[-l])
            delta = sd * np.dot(self.weights[-l+1].transpose(), delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def average_cost(self, data):
        norms = [LA.norm((self.feedforward(X)- y).reshape(10))
                 for X, y in data]
        return np.average(norms)
    
    # if the cost is very small, we can say that the model fits the training data
    def show_average_cost(self, data):
        cost = self.average_cost(data)
        print("current average cost {0}".format(cost))

    def set_check_cost_inside_SGD(self):
        self.check_cost_inside_SGD = True

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def save_model(self):
        joblib.dump(self, "my_model.pkl")
    

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def load_model():
    return joblib.load("my_model.pkl")

def train_model():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.fit(training_data)
    net.save_model()
    return net