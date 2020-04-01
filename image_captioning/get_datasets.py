#!/usr/bin/env python
# coding: utf-8


import csv
from shutil import copyfile
from pycocotools.coco import COCO

get_ipython().system('mkdir data')
get_ipython().system(
    'wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/')
get_ipython().system('unzip ./data/captions_train-val2014.zip -d ./data/')
get_ipython().system('rm ./data/captions_train-val2014.zip')

get_ipython().system('mkdir data/images')
get_ipython().system('mkdir data/images/train')
get_ipython().system('mkdir data/images/test')

coco = COCO('./data/annotations/captions_train2014.json')

# get ids of training images
with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)

trainIds = [int(i) for i in trainIds[0]]

for img_id in trainIds:
    path = coco.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/train2014/' + path, './data/images/train/' + path)

cocoTest = COCO('./data/annotations/captions_val2014.json')

with open('TestImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)

testIds = [int(i) for i in testIds[0]]

for img_id in testIds:
    path = cocoValTest.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/val2014/' + path, './data/images/test/' + path)
