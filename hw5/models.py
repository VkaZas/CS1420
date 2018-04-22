import numpy as np
import random


def l2_loss(Y, predictions):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y A 1D Numpy array with real values (float64)
        :param predictions A 1D Numpy array of the same size of Y
        :return L2 loss using predictions for Y.
    '''
    return np.power(np.linalg.norm(Y - predictions), 2)

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x A scalar or Numpy array
        :return Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x A scalar or Numpy array
        :return Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    return x


def linear_derivative(x):
    return 1


def step(x):
    return np.where(x > 0, 1, 0)


def step_derivative(x):
    return 0


def relu(x):
    return np.where(x > 0, x, 0)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


class LinearRegression:
    '''
        LinearRegression model that minimizes squared error using matrix inversion.
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the linear regression model.
        '''
        self.weights = None

    def train(self, X, Y):
        '''
        Trains the LinearRegression model by finding the optimal set of weights
        using matrix inversion.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return None
        '''
        self.weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        return np.matmul(X, np.transpose(self.weights))

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        self.weights = np.zeros(len(X[0]))

        for epoch in range(epochs):
            indices = np.random.permutation(len(Y))
            X = X[indices, :]
            Y = Y[indices]

            for i in range(len(Y)):
                x = X[i, :]
                y = Y[i]

                # Forward
                f = np.matmul(x, np.transpose(self.weights))

                # Backward
                d = (f - y) * x * 2
                self.weights = np.subtract(self.weights, d * learning_rate)

            if print_loss:
                print('Epoch %d loss: %f\n' % (epoch, self.average_loss(X, Y)))

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        return np.matmul(X, np.transpose(self.weights))

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

        # These are the learned parameters for the 2-Layer NN you will implement
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_weights = None
        self.output_bias = None

    def train(self, X, Y, learning_rate=0.01, epochs=1000, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :param learning_rate The learning rate to use for SGD
        :param epochs The number of times to pass through the dataset
        :param print_loss If True, print the loss after each epoch.
        :return None
        '''
        self.hidden_weights = np.random.normal(0, 0.01, (len(X[0]), self.hidden_size))
        self.hidden_bias = np.zeros(self.hidden_size)
        self.output_weights = np.random.normal(0, 0.01, self.hidden_size)
        self.output_bias = np.zeros(1)

        for epoch in range(epochs):
            indices = np.random.permutation(len(Y))
            X = X[indices, :]
            Y = Y[indices]

            for i in range(len(Y)):
                x = X[i, :]
                y = Y[i]

                # Forward
                h = np.matmul(x, self.hidden_weights) + self.hidden_bias
                f = np.matmul(self.activation(h), np.transpose(self.output_weights)) + self.output_bias

                # Backward
                db2 = 2 * (f - y)
                db1 = 2 * (f - y) * self.output_weights * self.activation_derivative(h)
                dv = 2 * (f - y) * self.activation(h)
                dw = np.zeros((len(X[0]), self.hidden_size))
                for k in range(len(X[0])):
                    for j in range(self.hidden_size):
                        dw[k, j] = 2 * (f - y) * self.output_weights[j] * x[k] * self.activation_derivative(h[j])

                self.hidden_weights = self.hidden_weights - dw * learning_rate
                self.output_weights = self.output_weights - dv * learning_rate
                self.hidden_bias -= db1 * learning_rate
                self.output_bias -= db2 * learning_rate

            if print_loss:
                print('Epoch %d loss: %f\n' % (epoch, self.average_loss(X, Y)))

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X 2D Numpy array where each row contains an example.
        :return A 1D Numpy array with one element for each row in X containing the predicted value.
        '''
        return np.array([np.matmul(sigmoid(np.matmul(x, self.hidden_weights) + self.hidden_bias), np.transpose(self.output_weights)) + self.output_bias for x in X]).flatten()

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
