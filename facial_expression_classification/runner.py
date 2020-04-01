from optimization import train
from model import LogisticRegression, SoftmaxRegression
import logging

# Constants
from constants import EMOTIONS, FOLDS, EPOCHS, NUM_COMPONENTS, PATH, MODEL, BATCH, LEARNING_RATE, GRAPH_STD, TASK
from graphing import graph_5c, graph_6b


def task_5_b():
    return {
        EMOTIONS: ['happiness', 'anger'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 5,
        PATH: "./resized",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 1,
        GRAPH_STD: False,
        TASK: "TASK5b"
    }


def task_5_c():
    return {
        EMOTIONS: ['happiness', 'anger'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 5,
        PATH: "./aligned",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 0.9,
        GRAPH_STD: True,
        TASK: "TASK5c"
    }


def task_5_c_low():
    return {
        EMOTIONS: ['happiness', 'anger'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 5,
        PATH: "./aligned",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 0.05,
        GRAPH_STD: True,
        TASK: "TASK5c_low"
    }


def task_5_c_mid():
    return {
        EMOTIONS: ['happiness', 'anger'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 5,
        PATH: "./aligned",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 0.9,
        GRAPH_STD: True,
        TASK: "TASK5c_mid"
    }


def task_5_c_high():
    return {
        EMOTIONS: ['happiness', 'anger'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 4,
        PATH: "./aligned",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 20,
        GRAPH_STD: True,
        TASK: "TASK5c_high"
    }


def task_5_c_all():
    settings1 = task_5_c_high()
    settings2 = task_5_c_low()
    settings3 = task_5_c_mid()

    loss1 = train(settings1)
    loss2 = train(settings2)
    loss3 = train(settings3)

    graph_5c(loss1, loss2, loss3, settings1, settings2, settings3)


def task_5_d():
    return {
        EMOTIONS: ['fear', 'surprise'],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 5,
        PATH: "./aligned",
        MODEL: LogisticRegression,
        BATCH: True,
        LEARNING_RATE: 0.9,
        GRAPH_STD: True,
        TASK: "TASK5d"
    }


def task_6_a():
    return {
        EMOTIONS: ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 10,
        PATH: "./aligned",
        MODEL: SoftmaxRegression,
        BATCH: True,
        LEARNING_RATE: 0.9,
        GRAPH_STD: True,
        TASK: "softtest"
    }


def task_6_b():
    settings1 = task_6_a()
    settings2 = {
        EMOTIONS: ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 10,
        PATH: "./aligned",
        MODEL: SoftmaxRegression,
        BATCH: False,
        LEARNING_RATE: 0.05,  # Look at comment in SGD in model.py for why we lower the learning rate.
        GRAPH_STD: True,
        TASK: "stochastic"
    }

    loss1 = train(settings1)
    loss2 = train(settings2)

    graph_6b(loss1, loss2, settings1, settings2)


def test():
    return {
        EMOTIONS: ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        FOLDS: 10,
        EPOCHS: 50,
        NUM_COMPONENTS: 10,
        PATH: "./aligned",
        MODEL: SoftmaxRegression,
        BATCH: False,
        LEARNING_RATE: 0.05,  # Look at comment in SGD in model.py for why we lower the learning rate.
        GRAPH_STD: True,
        TASK: "stochastic"
    }

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("./app.log"),
            logging.StreamHandler()
        ])

    # For each of the task we have specified some settings.
    # Remove the comment for the one you would like to run.

    #setting = task_5_b()
    # setting = task_5_c()
    # task_5_c_all()
    # setting = task_5_d()
    # setting = task_6_a()
    # task_6_b()
    settings = test()
    train(settings)
