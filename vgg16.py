import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import transform

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# TODO: need to implement batchnorm in the future.

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.bns = []

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.bns.append(self.bn_1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bns.append(self.bn_2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.bns.append(self.bn_3)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(512)
        self.bns.append(self.bn_4)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(512)
        self.bns.append(self.bn_5)



    def build(self, X):
        # transform pic into 3 pixel
        X = transform.resize(X, (224, 224, 3), preserve_range=True)
        X -= np.array([_R_MEAN, _G_MEAN, _B_MEAN])

        layer_out = []
        # TODO: there might something wrong, use batch norm layer followed by relu layer
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        bn_h = self.bns[0](h)
        layer_out.append(bn_h)
        # h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        bn_h = self.bns[1](h)
        layer_out.append(bn_h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        bn_h = self.bns[2](h)
        layer_out.append(bn_h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        bn_h = self.bns[3](h)
        layer_out.append(bn_h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5_1(self.bns[4](h)))
        h = F.relu(self.conv5_2(self.bns[4](h)))
        h = F.relu(self.conv5_3(self.bns[4](h)))
        bn_h = self.bns[4](h)
        layer_out.append(bn_h)
        # used as input for RNN
        return layer_out