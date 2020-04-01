import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show(img, epoch, config):
    npimg = img.numpy()
    plt.imsave(config['res_path'] + str(epoch) + ".jpg", np.transpose(npimg, (1, 2, 0)), format='jpg')
