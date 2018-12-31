import cv2
import os
import sys
import glob
from random import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import torch
import tables

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAINVAL_DATA_PATH = os.path.join(DATA_DIR, 'train')
HDF5_TRAIN_PATH = os.path.join(DATA_DIR, 'train_val_data.hdf5')

train_val_set = []

for label in os.listdir(TRAINVAL_DATA_PATH):
    paths = glob.glob(os.path.join(TRAINVAL_DATA_PATH, label, '*.ppm'))
    for path in paths:
        subset = (path, int(label))
        train_val_set.append(subset)

shuffle(train_val_set)
addrs, labels = zip(*train_val_set)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.8 * len(addrs))]
train_labels = labels[0:int(0.8 * len(labels))]

val_addrs = addrs[int(0.8 * len(addrs)):]
val_labels = labels[int(0.8 * len(labels)):]

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
data_shape = (0, 32, 32, 1)

# open a hdf5 file and create earrays
hdf5_file = tables.open_file(HDF5_TRAIN_PATH, mode='w')
train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_imgs', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_imgs', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)

# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape + (1,))
    # img = img.transpose(2, 0, 1)
    # save the image
    train_storage.append(img[None])

# loop over train addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Val data: {}/{}'.format(i, len(val_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape + (1,))
    # img = img.transpose(2, 0, 1)
    # save the image
    val_storage.append(img[None])

hdf5_file.close()
