import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from decision_tree import DecisionTree, train_error, entropy, gini_index


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    print('============================================================================')
    print('Dataset: ' + filename)
    print('---------------------------------')
    print('Gain function: train_error()')
    dtree = DecisionTree(train_data, gain_function=train_error)
    dtree_pruned = DecisionTree(train_data, validation_data, gain_function=train_error)
    print('Average training loss (not-pruned): ' + str(dtree.loss(train_data)))
    print('Average test loss (not-pruned): ' + str(dtree.loss(test_data)))
    print('Average training loss (pruned): ' + str(dtree_pruned.loss(train_data)))
    print('Average test loss (pruned): ' + str(dtree_pruned.loss(test_data)))

    print('---------------------------------')
    print('Gain function: entropy()')
    dtree = DecisionTree(train_data, gain_function=entropy)
    dtree_pruned = DecisionTree(train_data, validation_data, gain_function=entropy)
    print('Average training loss (not-pruned): ' + str(dtree.loss(train_data)))
    print('Average test loss (not-pruned): ' + str(dtree.loss(test_data)))
    print('Average training loss (pruned): ' + str(dtree_pruned.loss(train_data)))
    print('Average test loss (pruned): ' + str(dtree_pruned.loss(test_data)))

    print('---------------------------------')
    print('Gain function: gini_index()')
    dtree = DecisionTree(train_data, gain_function=gini_index)
    dtree_pruned = DecisionTree(train_data, validation_data, gain_function=gini_index)
    print('Average training loss (not-pruned): ' + str(dtree.loss(train_data)))
    print('Average test loss (not-pruned): ' + str(dtree.loss(test_data)))
    print('Average training loss (pruned): ' + str(dtree_pruned.loss(train_data)))
    print('Average test loss (pruned): ' + str(dtree_pruned.loss(test_data)))



    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
