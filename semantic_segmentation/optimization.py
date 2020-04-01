from utils import iou, iou_class, pixel_acc, inverted_weights
from plotting import generate_lineplot
from dataloader import *

# PyTorch
import torch.nn as nn
import torch
import torch.nn.functional as F

# Other
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
from functools import reduce
import logging


def init_weights(m):
    """
    Initialize weights of a layer in a neural network
    :param m: Layer in network
    :return: None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.uniform_(m.bias.data)


def train(model, optimizer, criterion, train_loader, val_loader, config):
    """
    Train model, graph and write results to csv
    :param model: Model to be trained
    :param optimizer: Optimizer used while training. E.g Adam
    :param criterion: Loss function used while training. E.g. CrossEntropy
    :param train_loader: Dataset of training images and labels
    :param val_loader: Dataset of validation images and labels
    :param config: Configuration for training loop
    :return: None
    """
    model.train()
    # Average Pixel Accuracy and average IoU (include only classes with trainId != 255) along with the IoU for the
    # classes - building(11), traffic sign(20), person(24), car(26), bicycle(33).
    train_losses = []
    val_losses = []
    val_ious = []
    val_accs = []

    building_ious = []
    sign_ious = []
    person_ious = []
    car_ious = []
    bike_ious = []
    epochs_completed = 0
    best_val_loss = 9999999

    for epoch in range(config['epochs']):
        ts = time.time()
        train_loss = 0
        n_batches = 0
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if config['use_gpu']:
                inputs = X.to('cuda')  # Move your inputs onto the gpu
                labels = Y.to('cuda')  # Move your labels onto the gpu
            else:
                inputs, labels = X, Y  # Unpack variables into inputs and labels
            outputs = model(inputs)

            # Check for transfer learning output
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            del inputs
            del labels
            torch.cuda.empty_cache()
            if iter % 10 == 0:
                logging.info("Epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches  # Average training loss over batches
        logging.info("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        class_ious, val_iou, val_acc, val_loss = val(model, val_loader, epoch, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'best_model.pth')

        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        building_ious.append(class_ious[11])
        sign_ious.append(class_ious[20])
        person_ious.append(class_ious[24])
        car_ious.append(class_ious[26])
        bike_ious.append(class_ious[33])
        model.train()
        epochs_completed += 1

        # check if errors in val_errors are strictly increasing
        LARGE_NUM = np.inf
        if len(val_losses) > 4 and reduce(lambda i, j: j if i < j else LARGE_NUM, val_losses[-5:]) != LARGE_NUM:
            logging.info("Early stopped at epoch {}".format(epoch))
            break

    # Generate graphs and gather results to csv file
    logging.info("Generating plot to baseline_train_val_loss.png")
    generate_lineplot((train_losses, val_losses), "Loss", "baseline_train_val_loss.png")

    logging.info("Building:".format(building_ious))
    logging.info("Sign:".format(sign_ious))
    logging.info("Person: {}".format(person_ious))
    logging.info("Car: {}".format(car_ious))
    logging.info("Bike: {}".format(bike_ious))
    logging.info("Val IOUS: {}".format(val_ious))
    logging.info("Val ACC: {}".format(val_accs))

    columns = ["Epoch", "Validation IOU", "Validation Accuracy", "Building IOU", "Traffic sign IOU",
               "Person IOU", "Car IOU", "Bicycle IOU"]
    df = pd.DataFrame(columns=columns)
    for i in range(epochs_completed):
        df.loc[len(df)] = [i, val_ious[i], val_accs[i], building_ious[i], sign_ious[i],
                           person_ious[i], car_ious[i], bike_ious[i]]
    logging.info("Writing results to res.csv")
    df.to_csv("res.csv")


def val(model, val_loader, epoch, config):
    """

    :param model: Model to use when predicting on validation data
    :param val_loader: Dataset of validation images and labels
    :param epoch: Current epoch number
    :param config: Configurations for the validation loop
    :return: avg_class_ious, avg_iou, avg_acc, avg_loss
    """
    logging.info("=====================================")
    logging.info("Validating at epoch {}".format(epoch))
    model.eval()
    ious = []
    accs = []
    loss = []

    # building(11), traffic sign(20), person(24), car(26), bicycle(33).
    class_ious = {
        11: [],
        20: [],
        24: [],
        26: [],
        33: []
    }
    with torch.no_grad():
        for iter, (X, tar, Y) in enumerate(val_loader):
            if config['use_gpu']:
                inputs = X.to('cuda')  # Move your inputs onto the gpu
                labels = Y.to('cuda')
            else:
                inputs, labels = X, Y

            outputs = model(inputs)

            # Check for transfer learning output
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # Take softmax and convert to predicted labels
            preds = F.softmax(outputs, dim=1)
            preds = torch.argmax(preds, 1)

            ious.append(iou(preds, labels))
            accs.append(pixel_acc(preds, labels))
            if config['weighted_loss']:
                loss.append(torch.nn.functional.cross_entropy(outputs, labels, weight=torch.from_numpy(inverted_weights).float().cuda()).cpu())
            else:
                loss.append(torch.nn.functional.cross_entropy(outputs, labels).cpu())
            for key in class_ious:
                class_ious[key].append(iou_class(preds, labels, key))

    avg_iou = np.average(np.stack(ious))
    avg_acc = np.average(np.stack(accs))
    avg_loss = np.average(np.stack(loss))
    avg_class_ious = dict()
    for key in class_ious:
        avg_class_ious[key] = np.average(np.stack(class_ious[key]))
    logging.info("Avg IOU: {}".format(avg_iou))
    logging.info("Avg acc: {}".format(avg_acc))
    logging.info("Avg loss: {}".format(avg_loss))
    logging.info("=====================================")
    return avg_class_ious, avg_iou, avg_acc, avg_loss


def test(test_loader, config):
    """
    Test the best model saved on the test dataset
    :param test_loader: Dataset containing test images
    :param config: Configurations for test loop
    :return:
    """
    model = torch.load('best_model.pth')
    model.eval()
    with torch.no_grad():
        for iter, (X, tar, Y) in enumerate(test_loader):
            if config['use_gpu']:
                inputs = X.to('cuda')  # Move your inputs onto the gpu
            else:
                inputs = X
            outputs = model(inputs)

            # Check for transfer learning output
            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            # Take softmax and convert to predicted labels
            preds = F.softmax(outputs, dim=1)
            preds = torch.argmax(preds, 1).cpu().numpy()
            stacked_img = np.stack((np.squeeze(preds),)*3)
            for i in range(preds.shape[1]):
                for j in range(preds.shape[2]):
                    pred_label = preds[0, i, j]
                    color = labels_dict[pred_label]['color']
                    stacked_img[:,i,j] = np.array(color)
            stacked_img = np.transpose(stacked_img, (1, 2, 0))
            # save labels as img
            matplotlib.image.imsave('test.png', stacked_img.astype("uint8"))
            break
