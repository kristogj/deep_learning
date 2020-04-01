################################################################################
# CSE 253: Programming Assignment 1
# Code snippet by Michael
# Winter 2020
################################################################################
# We've provided you with the dataset in PA1.zip
################################################################################
# To install PIL, refer to the instructions for your system:
# https://pillow.readthedocs.io/en/5.2.x/installation.html
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict

'''
list of face expressions (contempt, neutral are excluded) are:
1. anger
2. disgust
3. fear
4. happiness
5. sadness
6. surprise
'''


def load_data(data_dir="./aligned/"):
    """ Load all PNG images stored in your data directory into a list of NumPy
    arrays.

    Args:
        data_dir: The relative directory path to the CK+ image directory.
    Returns:
        images: A dictionary with keys as emotions and a list containing images associated with each key.
        cnt: A dictionary that stores the # of images in each emotion
    """
    images = defaultdict(list)

    # Get the list of emotional directory:
    for e in listdir(data_dir):
        # excluding any non-directory files
        if not os.path.isdir(os.path.join(data_dir, e)):
            continue
        # Get the list of image file names
        all_files = listdir(os.path.join(data_dir, e))

        for file in all_files:
            # Load only image files as PIL images and convert to NumPy arrays
            if '.png' in file:
                img = Image.open(os.path.join(data_dir, e, file))
                images[e].append(np.array(img))

    print("Emotions: {} \n".format(list(images.keys())))

    cnt = defaultdict(int)
    for e in images.keys():
        print("{}: {} # of images".format(e, len(images[e])))
        cnt[e] = len(images[e])
    return images, cnt


def balanced_sampler(dataset, cnt, emotions):
    # this ensures everyone has the same balanced subset for model training, don't change this seed value
    random.seed(20)
    print("\nBalanced Set:")
    min_cnt = min([cnt[e] for e in emotions])
    balanced_subset = defaultdict(list)
    for e in emotions:
        balanced_subset[e] = copy.deepcopy(dataset[e])
        random.shuffle(balanced_subset[e])
        balanced_subset[e] = balanced_subset[e][:min_cnt]
        print('{}: {} # of images'.format(e, len(balanced_subset[e])))
    return balanced_subset


def display_face(img):
    """ Display the input image and optionally save as a PNG.

    Args:
        img: The NumPy array or image to display

    Returns: None
    """
    # Convert img to PIL Image object (if it's an ndarray)
    if type(img) == np.ndarray:
        print("Converting from array to PIL Image")
        img = Image.fromarray(img)

    # Display the image
    img.show()


def cross_val(path="./aligned", k=5, emotions=['sadness', 'happiness']):
    """ k-fold cross validation on a given dataset.

    Args :
        k : number of folds to be performed
        emotions : list of emotions to use from the original dataset

    Returns : lists of the different sets (training, validation, test)
             for each of the folds with their labels.
    """

    im_dict, cnt = load_data(data_dir=path)
    balanced_set = balanced_sampler(im_dict, cnt, emotions)

    folds = []
    labels = []

    for i in range(k):
        folds.append([])
        labels.append([])

    # Creation of the different folds
    for i in emotions:
        elt = 0
        while elt < (len(balanced_set[i]) - 1):
            for x in range(k):
                if elt == len(balanced_set[i]):
                    break
                folds[x].append(balanced_set[i][elt])
                labels[x].append(emotions.index(i))
                elt += 1

    train, train_labels = [], []
    validation, validation_labels = [], []
    test, test_labels = [], []

    # Creation of the different combination of training, validation and test sets
    for elt in range(k):
        val, val_labels = folds[elt], labels[elt]
        test_set, test_set_labels = folds[(elt + 1) % k], labels[(elt + 1) % k]

        train_set = [folds[i] for i in range(len(folds)) if i != elt and i != (elt + 1) % k]
        train_set_labels = [labels[i] for i in range(len(labels)) if i != elt and i != (elt + 1) % k]

        tr_set = []
        for i in train_set:
            tr_set += i

        tr_labels = []
        for i in train_set_labels:
            tr_labels += i

        # the sets are stored in order in lists to be later used
        train.append(tr_set)
        train_labels.append(tr_labels)

        validation.append(val)
        validation_labels.append(val_labels)

        test.append(test_set)
        test_labels.append(test_set_labels)

    return train, train_labels, validation, validation_labels, test, test_labels