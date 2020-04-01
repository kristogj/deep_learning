"""
This file contains the objective function
"""
import numpy as np
from model import LogisticRegression, SoftmaxRegression


def logistic_loss_function(model: LogisticRegression, X: np.ndarray, targets: np.ndarray):
    """

    Args:
        model: Logistic regression model
        X: array of pca transformed images
        targets: actual labels for the images the predictions are done on

    Returns:
        The average cross entropy loss over all the predictions
    """
    predictions = model.predict(X)

    # Error when label equal 1
    loss_1 = targets * np.log(predictions)

    # Error when label equal 0
    loss_0 = (1 - targets) * np.log(1 - predictions)

    total_loss = loss_1 + loss_0

    # return the average loss overall
    return - total_loss.sum() / targets.shape[0]


def softmax_loss_function(model: SoftmaxRegression, X: np.ndarray, targets: np.ndarray):
    """

    Args:
        model: regression model
        X: array of pca transformed images
        targets: actual labels for the images the predictions are done on

    Returns:
        The average cross entropy loss over all the predictions
    """
    predictions = model.predict(X)

    sum_of_score = 0
    for n in range(X.shape[0]):
        for c in range(model.c):
            sum_of_score += targets[n, c] * np.log(predictions[n, c])

    return - sum_of_score / targets.shape[0]


def logistic_accuracy(model: LogisticRegression, X: np.ndarray, targets: np.ndarray):
    predictions = model.predict(X)  # These are probabilities
    predictions = np.around(predictions)
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    correct = sum(predictions == targets)
    return correct / len(targets)


def softmax_accuracy(model: SoftmaxRegression, X: np.ndarray, targets: np.ndarray):
    predictions = model.predict(X)  # These are probabilities
    predictions = predictions.argmax(axis=1)
    targets = targets.argmax(axis=1)
    correct = sum(predictions == targets)
    return correct / len(targets)
