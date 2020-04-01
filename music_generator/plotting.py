import matplotlib.pyplot as plt
import numpy as np


def save_loss_graph(model):
    """
    Save the models training and validation loss plot to file
    :param model:
    :return: None
    """
    x = np.arange(1, len(model.training_losses) + 1, 1)
    plt.plot(x, model.training_losses, label="train loss")
    plt.plot(x, model.validation_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.xticks(x)
    plt.title("Loss as a function of number of epochs")
    plt.legend()
    plt.savefig('loss-plot.png')
