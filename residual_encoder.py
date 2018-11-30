import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from vgg16 import Vgg16

# TODO: might need to add batch norm between each conv layer
class ResidualEncoder(torch.nn.Module):
    def __init__(self, out_channels=2):
        super(ResidualEncoder, self).__init__()
        # using a pretrained model vgg16
        self.resnet = Vgg16()

        # convolution only adjust the width and height size
        # the b_conv4 layer use 1*1 filter
        self.b_conv4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # assert(self.bn_4.get_shape().as_list()[1:] == [28, 28, 512])
        # assert(self.b_conv4.get_shape().as_list()[1:] == [28, 28, 256])

        # Backward upscale layer 4
        self.b_conv4_upscale = nn.Upsample(scale_factor=2)
        self.b_conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Backward upscale layer 3
        self.b_conv3_upscale = nn.Upsample(scale_factor=2)
        self.b_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        # Backward upscale layer 2
        self.b_conv2_upscale = nn.Upsample(scale_factor=2)
        self.b_conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Backward upscale bottom layer
        self.b_conv1_upscale = nn.Upsample(scale_factor=2)
        self.b_conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.b_conv_last = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, img):
        bn_ls = self.resnet.build(img)
        bn_4 = bn_ls[4]
        conv4 = F.relu(self.b_conv4(bn_4))

        # first upscale and add with vgg 56 * 56 * 256 layer output
        conv4 = self.b_conv4_upscale(conv4)
        bn_3 = bn_ls[3]
        conv3_input = torch.add(conv4, bn_3)

        # conv to RNN second layer
        conv3 = F.relu(self.b_conv(conv3_input))

        # second upscale and add with vgg 112 * 112 * 128 layer output
        conv3 = self.b_conv3_upscale(conv3)
        bn_2 = bn_ls[2]
        conv2_input = torch.add(conv3, bn_2)

        # conv to RNN third layer
        conv2 = F.relu(self.b_conv2(conv2_input))

        # third upscale and add with vgg 224 * 224 * 64 layer output
        conv2 = self.b_conv2_upscale(conv2)
        bn_1 = bn_ls[1]
        conv1_input = torch.add(conv2, bn_1)

        # conv to RNN fourth layer
        conv1 = F.relu(self.b_conv1(conv1_input))

        # fourth upscale and add with vgg 224 * 224 * 3 layer output
        conv1 = self.b_conv1_upscale(conv1)
        bn_0 = bn_ls[0]
        conv0_input = torch.add(conv1, bn_0)

        # conv to RNN bottom layer
        conv0 = F.relu(self.b_conv0(conv0_input))

        # conv to UV layer
        bottom_layer = self.b_conv_last(conv0)

        # use sigmoid for last layer
        bottom_layer = F.sigmoid(bottom_layer)

        return bottom_layer



