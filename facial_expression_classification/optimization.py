"""
This file contains all the code for training and optimization of the model
"""

import numpy as np
import logging

from dataloader import cross_val
from pca import PCA, show
from model import LogisticRegression, SoftmaxRegression
from objective import logistic_loss_function, softmax_loss_function, logistic_accuracy, softmax_accuracy
from constants import EMOTIONS, FOLDS, EPOCHS, NUM_COMPONENTS, PATH, MODEL, BATCH, LEARNING_RATE, GRAPH_STD, TASK
from graphing import visualize_weights, confusion_matrix, graph_acc, graph_loss, graph_5c, graph_cm
from utils import shuffle_data, one_hot_encode


def train(settings):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = cross_val(path=settings[PATH], k=settings[FOLDS],
                                                         emotions=settings[EMOTIONS])
    # Each fold will have a new model that is used on the test data one time. The results are stored here
    test_loss, test_acc = [], []

    # Save all the models, this way we can access their loss and accuracy stats
    models = []

    # List of confusion matrix for later task
    cms = []

    # For every fold, fit a new PCA, train new model, do validation, save the best model
    for k in range(settings[FOLDS]):
        Xtrain_k, ytrain_k, Xval_k, yval_k, Xtest_k, ytest_k = Xtrain[k], ytrain[k], Xval[k], yval[k], Xtest[k], ytest[
            k]

        # Shuffle so there is no pattern
        Xtrain_k, ytrain_k = shuffle_data(Xtrain_k, ytrain_k)
        Xval_k, yval_k = shuffle_data(Xval_k, yval_k)
        Xtest_k, ytest_k = shuffle_data(Xtest_k, ytest_k)

        logging.info("Started fold number: {}".format(k))

        # Convert to numpy arrays
        Xtrain_k, Xval_k, Xtest_k = np.array(Xtrain_k), np.array(Xval_k), np.array(Xtest_k)

        # Based on model there is different functions that needs to be set
        if settings[MODEL] == SoftmaxRegression:
            ytrain_k, yval_k, ytest_k = one_hot_encode(ytrain_k), one_hot_encode(yval_k), one_hot_encode(ytest_k)
            loss_function = softmax_loss_function
            accuracy = softmax_accuracy
        else:
            ytrain_k, yval_k, ytest_k = np.reshape(ytrain_k, (-1, 1)), np.reshape(yval_k, (-1, 1)), np.reshape(ytest_k,
                                                                                                               (-1, 1))
            loss_function = logistic_loss_function
            accuracy = logistic_accuracy
        # Fit the pca only using training data
        pca = PCA(settings[NUM_COMPONENTS])
        pca.fit(Xtrain_k)

        # Project Xtrain, Xval, Xtest onto the principal components
        Xtrain_k, Xval_k, Xtest_k = pca.transform(Xtrain_k), pca.transform(Xval_k), pca.transform(Xtest_k)

        # Make new model for this fold
        model = settings[MODEL](settings)

        best_weights, min_loss = model.weights, np.inf
        for epoch in range(1, settings[EPOCHS] + 1):
            # Select method for updating weights
            if settings[BATCH]:
                model.batch_gradient_descent(Xtrain_k, ytrain_k)
            else:
                model.stochastic_gradient_descent(Xtrain_k, ytrain_k)

            # Using an objective function to calculate the loss, and calculate the accuracy
            train_loss = loss_function(model, Xtrain_k, ytrain_k)
            val_loss = loss_function(model, Xval_k, yval_k)
            train_acc = accuracy(model, Xtrain_k, ytrain_k)
            val_acc = accuracy(model, Xval_k, yval_k)

            # Save the result for later graphs
            model.train_loss.append(train_loss)
            model.val_loss.append(val_loss)
            model.train_acc.append(train_acc)
            model.val_acc.append(val_acc)

            # Check if this is the lowest loss so far, if then save weights for best model
            if val_loss < min_loss:
                best_weights = np.copy(model.weights)
                min_loss = val_loss

            # Status update on how the training goes
            if epoch % 10 == 0:
                logging.info("Epoch: {}, Train_loss: {} , Val_loss: {}, Train_acc: {}, Val_acc: {}"
                             .format(epoch, train_loss, val_loss, train_acc, val_acc))

        # Now update the weights in the model to the best weights
        model.weights = best_weights

        # Use this model on the test data, and save loss & accuracy
        test_loss.append(loss_function(model, Xtest_k, ytest_k))
        test_acc.append(accuracy(model, Xtest_k, ytest_k))

        if settings[MODEL] == SoftmaxRegression:
            cf_matrix = confusion_matrix(model, Xtest_k, ytest_k)
            cms.append(cf_matrix)


        # Model finished, add it to list of models
        models.append(model)

    # Calculate the average test_loss and test_acc
    avg_test_loss, avg_test_acc = np.mean(test_loss), np.mean(test_acc)
    std_test_acc = np.std(test_acc)

    logging.info("Average Test Loss Overall Folds: {}".format(avg_test_loss))
    logging.info("Average Test Accuracy Overall Folds: {}".format(avg_test_acc))
    logging.info("Std Test Accuracy Overall Folds: {}".format(std_test_acc))
    logging.info("Generating plots")

    train_losses = [model.train_loss for model in models]
    val_losses = [model.val_loss for model in models]
    train_acces = [model.train_acc for model in models]
    val_acces = [model.val_acc for model in models]

    graph_loss(train_losses, val_losses, settings)
    graph_acc(train_acces, val_acces, settings)
    pca.display_pc(settings)

    # Visualize the cf matrix and weights for each emotion
    if settings[MODEL] == SoftmaxRegression:
        avg_cf_matrix = np.mean(cms, axis=0) # Take the average of all matrixes
        graph_cm(avg_cf_matrix, settings)
        visualize_weights(models, pca, settings)

    return train_losses
