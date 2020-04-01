"""
Contains some help functions used in the project
"""
import numpy as np
import random


def shuffle_data(X, y):
    """
    Args:
        X: array of features
        y: array of labels corresponding to the features in X

    Returns:
        shuffled array of features, array of corresponding labels
    """
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    X, y = zip(*zipped)
    return list(X), list(y)


def one_hot_encode(y):
    """

    Args:
        y: list of category labels

    Returns:
        one hot encoded numpy array
    """
    return np.eye(max(y) + 1)[y]


def sigmoid(a):
    return 1 / (1 + np.exp(-a))
