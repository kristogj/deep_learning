"""
This file keeps all the functions for doing all the graphing for the report
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from constants import EMOTIONS, FOLDS, EPOCHS, NUM_COMPONENTS, PATH, MODEL, BATCH, LEARNING_RATE, GRAPH_STD, TASK
from pca import show


def visualize_weights(models, pca, settings):
    # Pick random model
    rmodel = models[0]
    weights = rmodel.weights
    inverse = np.dot(weights, pca.principal_components.T)
    inverse = inverse.reshape((weights.shape[0], pca.img_dims[0], pca.img_dims[1]))

    # Concatenate images then show
    img = inverse[0]
    for x in range(1, inverse.shape[0]):
        img = np.concatenate((img, inverse[x]), axis=1)
    show(img, title="{}".format(settings[EMOTIONS]), save_path="display_weights.png")


def confusion_matrix(model, X, targets):
    # Go from one hot to category label
    predictions = model.predict(X)
    predictions = predictions.argmax(axis=1)
    targets = targets.argmax(axis=1)

    # Set up the confusion matrix
    cm = np.zeros((model.c, model.c))
    np.add.at(cm, [targets, predictions], 1)

    # Divide by number of times the category was chosen
    divider = cm.sum(axis=1).reshape(cm.shape[0], -1)
    cm /= divider
    return cm


def graph_cm(cm, settings):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=settings[EMOTIONS], yticklabels=settings[EMOTIONS])
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.title('Emotions confusion matrix')
    plt.savefig('./graphs/cm_batch{}_{}.png'.format(settings[BATCH], settings[TASK]))
    plt.show()


def graph_acc(train_acces, val_acces, settings):
    # Compute mean over the k folds
    avg_train_acc = np.mean(train_acces, axis=0) * 100
    avg_val_acc = np.mean(val_acces, axis=0) * 100

    # X-axis
    epochs = np.arange(1, len(avg_train_acc) + 1)

    plt.plot(epochs, avg_train_acc, label="Avg Training Accuracy")
    plt.plot(epochs, avg_val_acc, label="Avg Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Number of components: {}, Learning rate: {}".format(settings[NUM_COMPONENTS], settings[LEARNING_RATE]))
    plt.legend(loc="best")
    plt.savefig('./graphs/acc_{}_batch{}_{}.png'.format("_".join(settings[EMOTIONS]), int(settings[BATCH]),
                                                        settings[TASK]))
    plt.show()


def graph_loss(train_losses, val_losses, settings):
    # Compute mean over the k folds
    avg_train_losses = np.mean(train_losses, axis=0)
    avg_val_losses = np.mean(val_losses, axis=0)

    # X-axis
    epochs = np.arange(1, len(avg_train_losses) + 1)

    # Compute standard deviation over k folds
    std_train_losses = np.std(train_losses, axis=0)
    std_val_losses = np.std(val_losses, axis=0)

    # Ugly part
    temp = np.zeros(settings[EPOCHS])
    for x in range(0, 50, 10):
        temp[x] += 1
    temp[-1] = 1
    temp[0] = 0
    std_train_losses *= temp
    std_val_losses *= temp

    # Plot the graph and save
    if settings[GRAPH_STD]:
        plt.errorbar(epochs, avg_train_losses, std_train_losses, label="Avg Training Loss")
        plt.errorbar(epochs, avg_val_losses, std_val_losses, label="Avg Validation Loss")
    else:
        plt.errorbar(epochs, avg_train_losses, label="Training Loss")
        plt.errorbar(epochs, avg_val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Number of components: {}, Learning rate: {}".format(settings[NUM_COMPONENTS], settings[LEARNING_RATE]))
    plt.legend(loc="best")
    plt.savefig('./graphs/loss_{}_batch{}_{}.png'.format("_".join(settings[EMOTIONS]),
                                                         int(settings[BATCH]), settings[TASK]))
    plt.show()


def graph_5c(train_loss_high, train_loss_low, train_loss_mid, settings1, settings2, settings3):
    # Compute mean over the k folds
    avg_train_losses_high = np.mean(train_loss_high, axis=0)
    avg_train_losses_low = np.mean(train_loss_low, axis=0)
    avg_train_losses_mid = np.mean(train_loss_mid, axis=0)

    # X-axis
    epochs = np.arange(1, len(avg_train_losses_low) + 1)

    # Compute standard deviation over k folds
    std_train_losses_high = np.std(train_loss_high, axis=0)
    std_train_losses_low = np.std(train_loss_low, axis=0)
    std_train_losses_mid = np.std(train_loss_mid, axis=0)

    # Ugly part
    temp = np.zeros(settings1[EPOCHS])
    for x in range(0, 50, 10):
        temp[x] += 1
    temp[-1] = 1
    temp[0] = 0
    std_train_losses_high *= temp
    std_train_losses_low *= temp
    std_train_losses_mid *= temp

    # Plot the graph and save
    plt.errorbar(epochs, avg_train_losses_high, std_train_losses_high, label="Avg Training Loss for high learning rate")
    plt.errorbar(epochs, avg_train_losses_low, std_train_losses_low, label="Avg Training Loss for low learning rate")
    plt.errorbar(epochs, avg_train_losses_mid, std_train_losses_mid, label="Avg Training Loss for right learning rate")

    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Number of components: {}, Learning rates ({}, {}, {})".format(settings2[NUM_COMPONENTS],
                                                                             settings1[LEARNING_RATE],
                                                                             settings2[LEARNING_RATE],
                                                                             settings3[LEARNING_RATE]))
    plt.legend(loc="best")
    plt.savefig('./graphs/loss_{}_batch{}_TASK5c_all.png'.format("_".join(settings1[EMOTIONS]),
                                                                 int(settings1[BATCH])))
    plt.show()


def graph_6b(train_loss_batch, train_loss_stoc, settings1, settings2):
    # Compute mean over the k folds
    avg_train_losses_batch = np.mean(train_loss_batch, axis=0)
    avg_train_losses_stoc = np.mean(train_loss_stoc, axis=0)

    # Compute standard deviation over k folds
    std_train_losses_batch = np.std(train_loss_batch, axis=0)
    std_train_losses_stoc = np.std(train_loss_stoc, axis=0)

    # Ugly part
    temp = np.zeros(settings1[EPOCHS])
    for x in range(0, 50, 10):
        temp[x] += 1
    temp[-1] = 1
    temp[0] = 0
    std_train_losses_batch *= temp
    std_train_losses_stoc *= temp

    # X-axis
    epochs = np.arange(1, len(avg_train_losses_batch) + 1)

    plt.errorbar(epochs, avg_train_losses_batch, std_train_losses_batch, label="Avg Training Loss for batch gradient descent")
    plt.errorbar(epochs, avg_train_losses_stoc, std_train_losses_stoc, label="Avg Training Loss for stochastic gradient descent")

    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title(
        "Number of components: {}, Learning rate {}".format(settings1[NUM_COMPONENTS], settings1[LEARNING_RATE]))
    plt.legend(loc="best")
    plt.savefig('./graphs/batch_vs_stoch_train_loss.png')
    plt.show()
