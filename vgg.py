from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, \
            names=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']):
        super(Vgg16, self).__init__()
        self.names = names
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = vgg_pretrained_features[:3] #conv1_2
        self.slice2 = vgg_pretrained_features[3:8] #conv2_2
        self.slice3 = vgg_pretrained_features[8:13] #conv3_2
        self.slice4 = vgg_pretrained_features[13:20] #conv4_2
        self.slice5 = vgg_pretrained_features[20:27] #conv5_2
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_conv1_2 = h
        h = self.slice2(h)
        h_conv2_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        h = self.slice5(h)
        h_relu5_2 = h
        vgg_outputs = namedtuple("VggOutputs", self.names)
        out = vgg_outputs(h_conv1_2, h_conv2_2, h_relu3_2, h_relu4_2, h_relu5_2)
        return out