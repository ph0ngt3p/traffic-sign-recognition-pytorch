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


class MyTrainDataset(Dataset):
    def __init__(self, hdf5_file, root_dir, train, transform=None):
        if os.path.isfile(hdf5_file):
            self.hdf5_file = tables.open_file(hdf5_file, mode='r')
            self.train = train
            if train:
                self.train_imgs = self.hdf5_file.root.train_imgs
                self.train_labels = self.hdf5_file.root.train_labels
            else:
                self.val_imgs = self.hdf5_file.root.val_imgs
                self.val_labels = self.hdf5_file.root.val_labels
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

