import numpy as np
import sys
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import LinearRegression, OneLayerNN, TwoLayerNN
from models import linear, linear_derivative, step, step_derivative, relu, relu_derivative



def test_models(dataset, epochs, test_size=0.2):
    '''
        Tests LinearRegression, OneLayerNN, TwoLayerNN on a given dataset.

        :param dataset The path to the dataset
        :return None
    '''

    # Check if the file exists
    if not os.path.exists(dataset):
        print('The file {} does not exist'.format(dataset))
        exit()

    # Load in the dataset
    data = np.loadtxt(dataset, skiprows = 1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the features
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    print('Running models on {} dataset'.format(dataset))

    #### Linear Regression ######
    print('----- LINEAR REGRESSION -----')
    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b =np.append(X_test, np.ones((len(X_test), 1)), axis=1)
    regmodel = LinearRegression()
    regmodel.train(X_train_b, Y_train)
    print('Average Training Loss:', regmodel.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', regmodel.average_loss(X_test_b, Y_test))

    #### 1-Layer NN ######
    print('----- 1-Layer NN -----')
    nnmodel = OneLayerNN()
    nnmodel.train(X_train_b, Y_train, epochs=epochs, print_loss=False)
    print('Average Training Loss:', nnmodel.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', nnmodel.average_loss(X_test_b, Y_test))

    #### 2-Layer NN ######
    print('----- 2-Layer NN -----')
    model = TwoLayerNN(5)
    # Use X without a bias, since we learn a bias in the 2 layer NN.
    model.train(X_train, Y_train, epochs=epochs, print_loss=False)
    print('Average Training Loss:', model.average_loss(X_train, Y_train))
    print('Average Testing Loss:', model.average_loss(X_test, Y_test))

def main():

    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0)
    np.random.seed(0)
    test_models('data/wine.txt', 20)


if __name__ == "__main__":
    main()
