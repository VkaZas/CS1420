#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains two classifiers: Naive Bayes and Logistic Regression

   Brown CS142, Spring 2018
"""
import random

import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        self.p_cond = None
        self.p = None
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        X = data[0]
        Y = data[1]
        data_size, feat_size = X.shape

        # P_hat, the module itself
        self.p_cond = np.zeros((feat_size, self.n_classes))
        Y_cnt = np.unique(Y, return_counts=True)[1]
        # P
        self.p = Y_cnt / float(data_size)

        for i in range(self.n_classes):
            X_i = X[Y == i]

            for j in range(feat_size):
                self.p_cond[j, i] = np.count_nonzero(X_i[:, j]) / float(len(X_i))



    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """

        p_cond_pred = np.zeros(self.n_classes)

        for i in range(len(inputs)):
            p_cond_pred += np.log(self.p_cond[i] + 1e-5) if inputs[i] == 1 else np.log(1 - self.p_cond[i] + 1e-5)

        p_pred = self.p * p_cond_pred

        return p_pred.argmax()

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        X = data[0]
        Y = data[1]
        cnt = 0

        for i in range(len(Y)):
            cnt += int(self.predict(X[i]) == Y[i])

        return cnt / float(len(Y))

class LogisticRegression(object):
    """ Multinomial Linear Regression

    @attrs:
        weights: a parameter of the model
        alpha: the step size in gradient descent
        n_features: the number of features
        n_classes: the number of classes
    """
    def __init__(self, n_features, n_classes, epoch=100, decay=0.99):
        """ Initializes a LogisticRegression classifer. """
        self.alpha = 0.1  # tune this parameter
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes))
        self.epoch = epoch
        self.decay = decay

    def train(self, data, data_test = None):
        """ Trains the model, using stochastic gradient descent

        @params:
            data: a namedtuple including training data and its information
        @return:
            None
        """
        X = data[0]
        Y = data[1]

        # losses = []

        for i in range(self.epoch):
            select_idx = np.random.choice(len(Y), size=int(len(Y)/60.0))
            X_epoch = X[select_idx]
            Y_epoch = Y[select_idx]

            for j in range(len(Y_epoch)):
                y = Y_epoch[j]
                x = X_epoch[j]
                loss = self.calc_single_loss(x, y)

                # Adjust weight
                gradient = np.dot(X_epoch[j].reshape(self.n_features, 1), loss.reshape(1, self.n_classes))
                self.weights -= self.alpha * gradient

            self.alpha *= self.decay

            # losses.append(self.calc_total_loss(data_test[0], data_test[1]))
            # print('epoch:' + str(i))

        # fig, ax = plt.subplots(1)
        # ax.plot(losses, label='loss')
        # ax.set(xlabel='iteration', ylabel='loss norm', title='loss norm change along iteration')
        # ax.legend()
        # plt.show()
        #
        # print(losses)

    def calc_single_loss(self, x, y):
        y_test = self._softmax(np.dot(x, self.weights))

        # Calc loss
        loss = np.zeros(self.n_classes)

        for k in range(self.n_classes):
            if k == y:
                loss[k] += y_test[k] - 1
            else:
                loss[k] = y_test[k]

        return loss

    def calc_total_loss(self, X, Y):
        total_loss = 0
        for i in range(len(Y)):
            x = X[i]
            y = Y[i]
            loss = self.calc_single_loss(x, y)
            total_loss += np.linalg.norm(loss)

        return total_loss

    def predict(self, inputs):
        """ Compute predictions based on the learned parameters

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            a numpy array of predictions
        """
        y_test = self._softmax(np.dot(inputs, self.weights))
        return y_test.argmax()

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        X = data[0]
        Y = data[1]
        cnt = 0

        for i in range(len(Y)):
            cnt += int(self.predict(X[i]) == Y[i])

        return cnt / float(len(Y))

    def _softmax(self, x):
        """ apply softmax to an array

        @params:
            x: the original array
        @return:
            an array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
