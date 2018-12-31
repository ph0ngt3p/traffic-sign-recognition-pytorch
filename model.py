import os
import sys
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Net1(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 108, 3)
        self.conv2 = nn.Conv2d(108, 200, 3)
        self.fc1 = nn.Linear(6 * 6 * 200, 100)
        self.fc2 = nn.Linear(100, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net2(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 108, 3)
        self.conv2 = nn.Conv2d(108, 200, 3)
        self.fc1 = nn.Linear(15 * 15 * 108 + 6 * 6 * 200, 100)
        self.fc2 = nn.Linear(100, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), 2)
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        out = [x1, x2]
        out = torch.cat(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return F.log_softmax(out, dim=1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test():
    net = Net2(num_classes=43)
    y = net(torch.randn(1, 1, 32, 32))
    print(y.size())


# test()
