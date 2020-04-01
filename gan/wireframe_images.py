"""
Code copied from:
https://www.freecodecamp.org/news/sketchify-turn-any-image-into-a-pencil-sketch-with-10-lines-of-code-cf67fa4f68ce/
"""

import imageio
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

data_path = "./datasets/pokemon/"


def convert(data_path):
    try:
        os.makedirs(data_path + "trainA/")  # sketch
    except FileExistsError:
        pass

    try:
        os.makedirs(data_path + "trainB/")  # Real
    except FileExistsError:
        pass

    def grayscale(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def dodge(front, back):
        result = front * 255 / (255 - back)
        result[result > 255] = 255
        result[back == 255] = 255
        return result.astype('uint8')

    count = 0
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(".jpg"):
            start_img = imageio.imread(data_path + filename)

            # 1. Grayscale image
            gray_img = grayscale(start_img)

            # 2. Invert image
            inverted_img = 255 - gray_img

            # 3. Blur image
            blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=5)

            # 4. Color Dodge
            final_img = dodge(blur_img, gray_img)

            plt.imsave(data_path + "trainA/" + str(count) + "_A.jpg", final_img, cmap='gray', vmin=0, vmax=255)
            plt.imsave(data_path + "trainB/" + str(count) + "_B.jpg", start_img, vmin=0, vmax=255)

            count += 1


convert(data_path)
