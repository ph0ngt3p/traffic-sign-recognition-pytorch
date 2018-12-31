import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import tables
import numpy as np
from augmentations import rand_similarity_trans

import cv2
import matplotlib.pyplot as plt


class MyTrainDataset(Dataset):
    def __init__(self, hdf5_file, root_dir, train, aug_multiplier=5, transform=None):
        if os.path.isfile(hdf5_file):
            self.hdf5_file = tables.open_file(hdf5_file, mode='r')
            self.train = train
            if train:
                # Generate additional data
                print('Generating {} times more additional data for train set...'.format(aug_multiplier))
                orig_train_labels = self.hdf5_file.root.train_labels
                orig_train_imgs = self.hdf5_file.root.train_imgs
                # print(orig_imgs.shape)
                aug_train_imgs = np.vstack([rand_similarity_trans(img, aug_multiplier) for img in orig_train_imgs])
                # print(aug_imgs.shape)
                aug_train_labels = np.repeat(orig_train_labels, aug_multiplier)

                # append the generated data to the training data
                self.train_imgs = np.append(orig_train_imgs, aug_train_imgs, axis=0)
                self.train_labels = np.append(orig_train_labels, aug_train_labels, axis=0)
            else:
                print('Generating {} times more additional data for validation set...'.format(aug_multiplier))
                orig_val_labels = self.hdf5_file.root.val_labels
                orig_val_imgs = self.hdf5_file.root.val_imgs
                aug_val_imgs = np.vstack([rand_similarity_trans(img, aug_multiplier) for img in orig_val_imgs])
                aug_val_labels = np.repeat(orig_val_labels, aug_multiplier)

                # append the generated data to the training data
                self.val_imgs = np.append(orig_val_imgs, aug_val_imgs, axis=0)
                self.val_labels = np.append(orig_val_labels, aug_val_labels, axis=0)
            self.root_dir = root_dir
            self.transform = transform
        else:
            print('Data path is not available!')
            exit(1)

    def __len__(self):
        if self.train:
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)

    def __getitem__(self, idx):
        if self.train:
            image = self.train_imgs[idx]
            label = self.train_labels[idx]
        else:
            image = self.val_imgs[idx]
            label = self.val_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class MyTestDataset(Dataset):
    def __init__(self, hdf5_file, root_dir, transform=None):
        if os.path.isfile(hdf5_file):
            self.hdf5_file = tables.open_file(hdf5_file, mode='r')
            self.test_imgs = self.hdf5_file.root.test_imgs
            self.test_ids = self.hdf5_file.root.test_labels
            self.root_dir = root_dir
            self.transform = transform
        else:
            print('Data path is not available!')
            exit(1)

    def __len__(self):
        return len(self.test_imgs)

    def __getitem__(self, idx):
        image = self.test_imgs[idx]
        image_id = self.test_ids[idx]

        if self.transform:
            image = self.transform(image)

        return image, image_id


def gen_outputline(label, preds):
    idx = str(label)
    return idx + ',' + str(preds)[1:-1].replace(',', '') + '\n'


# def test_aug_train():
#     dset = MyTrainDataset('data/train_val_data.hdf5', root_dir='./data', train=True)
#     print(len(dset))
#     print(len(dset.train_labels))
#     print(dset.train_imgs.shape)
#
#     img = dset.train_imgs[-555, :, :, 0]
#     print(dset.train_labels[-555])
#     plt.subplot(1, 2, 2)
#     plt.imshow(img, cmap='gray')
#     plt.show()
#
#
# test_aug_train()
