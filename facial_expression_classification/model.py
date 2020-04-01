"""
This file contains the model classes
"""
import numpy as np
from constants import EMOTIONS, NUM_COMPONENTS, LEARNING_RATE
from utils import sigmoid


class LogisticRegression:

    def __init__(self, settings):
        self.learning_rate = settings[LEARNING_RATE]
        self.weights = np.ones((1, settings[NUM_COMPONENTS]))

        self.train_loss = []
        self.train_acc = []

        self.val_loss = []
        self.val_acc = []

    def predict(self, X):
        """

        Args:
            X: array of pca transformed images

        Returns:
            array of probabilities
        """
        a = np.dot(X, self.weights.T)
        return sigmoid(a)

    def batch_gradient_descent(self, X, targets):
        """
        Function to update the weights in one epoch of training
        Args:
            X: array of pca transformed images
            targets: array of labels for these images

        Returns:
            None
        """
        predictions = self.predict(X)

        # Compute the error between predictions and actual targets
        error = targets - predictions

        # Compute the gradients
        gradients = - np.dot(error.T, X)

        # Take the average
        gradients /= len(targets)

        # Update weights
        self.weights -= self.learning_rate * gradients


class SoftmaxRegression():
    def __init__(self, settings):
        self.c = len(settings[EMOTIONS])
        self.learning_rate = settings[LEARNING_RATE]
        self.weights = np.ones((self.c, settings[NUM_COMPONENTS]))

        self.train_loss = []
        self.train_acc = []

        self.val_loss = []
        self.val_acc = []

    def predict(self, X):
        """

        Args:
            X: array of pca transformed images

        Returns:
            array of probabilities
        """
        a = np.dot(X, self.weights.T)
        a = np.exp(a) / np.sum(np.exp(a), axis=len(a.shape) - 1, keepdims=True)
        return a

    def batch_gradient_descent(self, X, targets):
        """
        Function to update the weights in one epoch of training
        Args:
            X: array of pca transformed images
            targets: array of labels for these images

        Returns:
            None
        """
        predictions = self.predict(X)

        # Compute the error between predictions and actual targets
        error = targets - predictions

        # Compute the gradients
        gradients = - np.dot(error.T, X)

        # Take the average
        gradients /= len(targets)

        # Update weights
        self.weights -= self.learning_rate * gradients

    def stochastic_gradient_descent(self, X: np.ndarray, targets: np.ndarray):
        """
        Function to update the weights in one epoch of training
        Args:
            X: array of pca transformed images
            targets: array of labels for these images

        Returns:
            None
        """
        indices = list(range(len(targets)))
        np.random.shuffle(indices)
        for i in indices:
            image, target = X[[i], :], targets[[i], :]
            prediction = self.predict(image)

            # Compute the error between the prediction and the actual target
            error = target - prediction
            
            # Compute the gradient
            gradient = - np.dot(error.T, image)

            # Take the average - We are not doing that part for sgd, so use a lower lr.
            # gradients /= len(targets)

            # Update the weights
            self.weights -= self.learning_rate * gradient


if __name__ == '__main__':
    settings = {
        EMOTIONS: ['happiness', 'anger'],
        NUM_COMPONENTS: 5,
        LEARNING_RATE: 1,
    }
    SoftmaxRegression(settings)
