import csv
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from svm import SVM, linear_kernel, polynomial_kernel, rbf_kernel


def test_svm(train_data, test_data, kernel_func=linear_kernel, lambda_param=.1):
    """
    Create an SVM classifier with a specificied kernel_func, train it with
    train_data and print the accuracy of model on test_data

    :param train_data: a namedtuple including training inputs and training labels
    :param test_data: a namedtuple including test inputs and test labels
    :param kernel_func: kernel function to use in the SVM
    :return: None
    """
    svm_model = SVM(kernel_func=kernel_func, lambda_param=lambda_param)
    svm_model.train(train_data.inputs, train_data.labels)
    train_accuracy = svm_model.accuracy(train_data.inputs, train_data.labels)
    test_accuracy = svm_model.accuracy(test_data.inputs, test_data.labels)
    if not (train_accuracy is None):
        print('Train accuracy: ', round(train_accuracy * 100, 2), '%')
    else:
        print('Train accuracy: accuracy function not yet implemented.')
    if not (test_accuracy is None):
        print('Test accuracy:', round(test_accuracy * 100,2), '%')
    else:
        print('Test accuracy: accuracy function not yet implemented.')

    return train_accuracy, test_accuracy


def tune_polynomial_svm(train_data, test_data):
    train_acc = []
    test_acc = []
    for d in range(1, 9):
        poly_func = lambda x, y: polynomial_kernel(x, y, 0, d)
        a1, a2 = test_svm(train_data, test_data, kernel_func=poly_func)
        train_acc.append(a1)
        test_acc.append(a2)

    plt.title('Tuning d in polynomial kernel')
    plt.xlabel('d')
    plt.ylabel('acc')
    plt.plot(range(1, 9), train_acc, 'r', label='train_acc')
    plt.plot(range(1, 9), test_acc, 'b', label='test_acc')
    plt.legend()
    plt.grid()
    plt.show()


def tune_rbf_svm(train_data, test_data):
    train_acc = []
    test_acc = []
    m = train_data.inputs.shape[1]
    gammas = np.linspace(.5/m, 1.2/m, 10)
    for gamma in gammas:
        rbf_func = lambda x, y: rbf_kernel(x, y, gamma)
        a1, a2 = test_svm(train_data, test_data, kernel_func=rbf_func)
        train_acc.append(a1)
        test_acc.append(a2)

    plt.title('Tuning gammar in rbf kernel')
    plt.xlabel('gammar')
    plt.ylabel('acc')
    plt.plot(gammas, train_acc, 'r', label='train_acc')
    plt.plot(gammas, test_acc, 'b', label='test_acc')
    plt.legend()
    plt.grid()
    plt.show()


def plot_boundary_svm(train_data, title, kernel_func=linear_kernel, lambda_param=.1):
    svm_model = SVM(kernel_func=kernel_func, lambda_param=lambda_param)
    svm_model.train(train_data.inputs, train_data.labels)

    intv = .02
    X = train_data.inputs
    Y = train_data.labels * 2 - 1
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, intv), np.arange(x2_min, x2_max, intv))

    Z = svm_model.predict(np.c_[x1.ravel(), x2.ravel()])
    Z = Z.reshape(x1.shape)
    plt.contourf(x1, x2, Z, cmap=plt.cm.coolwarm)

    positive_idx = [i for i, y in enumerate(Y) if y == 1]
    negative_idx = [i for i, y in enumerate(Y) if y == -1]
    plt.scatter(X[positive_idx, 0], X[positive_idx, 1], c='g', marker='o', cmap=plt.cm.coolwarm)
    plt.scatter(X[negative_idx, 0], X[negative_idx, 1], c='r', marker='x', cmap=plt.cm.coolwarm)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()

def read_data(file_name):
    """
    Reads the data from the input file and splits it into inputs and labels

    :param file_name: path to the desired data file
    :return: two numpy arrays, one containing the inputs and one containing
              the labels
    """
    inputs, labels, classes = [], [], set()
    with open(file_name) as f:
        positive_label = None
        reader = csv.reader(f)
        for row in reader:
            example = np.array(row)
            classes.add(example[-1])
            if positive_label is None:
                positive_label = example[-1]
            label = 1 if example[-1] == positive_label else 0
            row.pop()
            labels.append(label)
            inputs.append([float(val) for val in row])

    if len(classes) > 2:
        print('Only binary classification tasks are supported.')
        exit()

    inputs = np.array(inputs)
    labels = np.array(labels)

    # Normalize the feature values
    for j in range(inputs.shape[1]):
        col = inputs[:,j]
        mu = np.mean(col)
        sigma = np.std(col)
        if sigma == 0: sigma = 1
        inputs[:,j] = 1/sigma * (col - mu)

    return inputs, labels

def main():
    """
    Reads in the data, trains an SVM model and outputs the training and
    testing accuracy of the model on the dataset. 
    """

    random.seed(0)
    np.random.seed(0)
    if len(sys.argv) != 2:
        print('Incorrect number of argments. Usage: python main.py <path_to_dataset>')
        exit()

    _, filename = sys.argv

    Dataset = namedtuple('Dataset', ['inputs', 'labels'])
    # Read data
    inputs, labels = read_data(filename)
    # Split data into training set and test set with a ratio of 2:1
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.20)


    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs[:], test_labels[:])

    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    n = train_data.inputs.shape[0]
    m = train_data.inputs.shape[1]

    # Set lambda parameter. You do not need to change this but are free to experiment.
    lambda_param = 1.0/(2*n)

    print('================ No kernel  =================')
    test_svm(train_data, test_data, kernel_func=linear_kernel, lambda_param=lambda_param)
    print('================ RBF kernel =================')
    # Set gamma to 1/m. This matches the behavior of sklearn's implementation.
    rbf_with_gamma = lambda x, y: rbf_kernel(x, y, 1.0/m)
    test_svm(train_data, test_data, kernel_func=rbf_with_gamma, lambda_param=lambda_param)
    print('============= Polynomial kernel =============')
    test_svm(train_data, test_data, kernel_func=polynomial_kernel, lambda_param=lambda_param)

    # plot_boundary_svm(train_data, 'linear kernel', kernel_func=linear_kernel, lambda_param=lambda_param)
    # plot_boundary_svm(train_data, 'polynomial kernel(d=2)', kernel_func=polynomial_kernel, lambda_param=lambda_param)
    # plot_boundary_svm(train_data, 'rbf kernel', kernel_func=rbf_kernel, lambda_param=lambda_param)

    # tune_polynomial_svm(train_data, test_data)
    # tune_rbf_svm(train_data, test_data)

    return

if __name__ == '__main__':
    main()
