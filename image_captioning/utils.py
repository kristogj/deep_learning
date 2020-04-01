import csv
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import logging

from pycocotools.coco import COCO


def get_ids(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        trainIds = list(reader)

    return [int(i) for i in trainIds[0]]


def get_anns(img_ids, path):
    img_ids = set(img_ids)
    ann_ids = list()

    coco = COCO(path)
    for ann_id in coco.anns:
        ann = coco.anns[ann_id]
        if ann['image_id'] in img_ids:
            ann_ids.append(ann_id)

    return ann_ids


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_val_sampler(dataset, random_seed, validation_split=.2, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def get_pretrained_embeddings(file_path, vocab):
    # Read weights per word to a dictionary
    file = open(file_path, "r")
    word2weight = dict()
    for line in file:
        line = line.split()
        word = line[0]
        weights = list(map(float, line[1:]))
        word2weight[word] = weights
    file.close()

    words_in_vocab = vocab.word_to_id.keys()

    # Find matching words between vocab and pre-trained and save their weights. Else random weights
    embedding_dim = 50  # Must fit the dimension from the pre-trained file
    pretrained_weight_matrix = np.zeros((len(words_in_vocab), embedding_dim))
    words_found = 0
    for i, word in enumerate(words_in_vocab):
        try:
            pretrained_weight_matrix[i] = word2weight[word]
            words_found += 1
        except KeyError:
            pretrained_weight_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    logging.info("Pre-trained Embedding: {} out of {} words matched".format(words_found, len(vocab)))

    # Convert to tensor and return
    return torch.from_numpy(pretrained_weight_matrix).type(torch.cuda.FloatTensor)
