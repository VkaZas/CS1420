"""
    This is a file you will have to fill in.

    It contains helper functions required by K-means method via iterative improvement

"""
import numpy as np
from kneighbors import euclidean_distance


def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids

    :param k: number of cluster centroids, an int
    :param inputs: a 2D Python list, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    return inputs[np.random.choice(len(inputs), k, replace=False)]


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance

    :param inputs: inputs of data, a 2D Python list
    :param centroids: a Numpy array of k current centroids
    :return: a Python list of centroid indices, one for each row of the inputs
    """
    # TODO
    res = []

    for i in range(len(inputs)):
        dists = [euclidean_distance(inputs[i], centroid) for centroid in centroids]
        res.append(np.argmin(dists))

    return res

def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster - the average of all data points in the cluster

    :param inputs: inputs of data, a 2D Python list
    :param indices: a Python list of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    dim = len(inputs[0])
    centroids = [np.zeros(dim) for i in range(k)]

    for i in range(len(indices)):
        centroids[indices[i]] += inputs[i]

    (values, counts) = np.unique(indices, return_counts=True)
    for i in range(len(centroids)):
        centroids[i] /= counts[i]

    return centroids


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    Use init_centroids, assign_step, and update_step!
    The only computation that should occur within this function is checking 
    for convergence - everything else should be handled by helpers

    :param inputs: inputs of data, a 2D Python list
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = init_centroids(k, inputs)

    for i in range(max_iter):
        indices = assign_step(inputs, centroids)
        new_centroids = update_step(inputs, indices, k)

        converge = True
        for j in range(len(new_centroids)):
            if euclidean_distance(new_centroids[j], centroids[j]) > tol:
                converge = False
                break

        if converge:
            # print('Converge at iteration' + str(i))
            break

        centroids = new_centroids

    return centroids
