import cv2
import os
import sys
import glob
from random import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import tables

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')
TEST_CSV_PATH = os.path.join(TEST_DATA_PATH, 'GT-final_test.csv')
HDF5_TEST_PATH = os.path.join(DATA_DIR, 'test_data.hdf5')

names = 'Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId'.split(';')
df = pd.read_csv(TEST_CSV_PATH, delimiter=';', names=names)

files = []
test_set = []

for file in list(df['Filename']):
    path = os.path.join(TEST_DATA_PATH, file)
    files.append(path)

test_set = list(zip(files, list(df['ClassId'])))
addrs, labels = zip(*test_set)

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
data_shape = (0, 32, 32, 1)

# open a hdf5 file and create earrays
hdf5_file = tables.open_file(HDF5_TEST_PATH, mode='w')
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_imgs', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_file.create_array(hdf5_file.root, 'test_labels', labels)

for i in range(len(addrs)):
    if i % 100 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_set)))
    addr = addrs[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape + (1, ))
    # img = img.transpose(2, 0, 1)
    test_storage.append(img[None])

hdf5_file.close()
