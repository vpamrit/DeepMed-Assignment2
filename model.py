#code inspired from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from torch.autograd import Variable
## each image is 490 x 326

#create the model classes

#how to initialize weights function
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

#this is just a layer where you pass in a lambda and it does the rest
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

#basic block with shortcut (not a bottleneck block)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', dropout=0.25):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                top = (int) ((self.expansion*planes - in_planes) / 2)
                bot = (self.expansion*planes - in_planes) - top
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, top, bot), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

##bottleneck block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option='B', dropout=0.25):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                top = (int) ((self.expansion*planes - in_planes) / 2)
                bot = (self.expansion*planes - in_planes) - top
                self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, top, bot), "constant", 0))
            elif option == 'B':
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion*planes)
                    )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7, option='B', dropout=0.25):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=3, option=option, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, option=option, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=3, option=option, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, option=option, dropout=dropout)

        self.linear = nn.Linear(98304, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, option, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

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


def ResNet18(dropout=0.25, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, dropout=dropout)

def ResNet34(dropout=0.25, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, dropout=dropout)

def ResNet50(dropout=0.25 num_classes=7):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, dropout=dropout)

def ResNet101(dropout=0.25, num_classes=7):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, dropout=dropout)

def ResNet152(dropout=0.25, num_classes=7):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, dropout=dropout)

def ResNet101(dropout=0.25, num_classes=7):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, dropout=dropout)

def ResNet152(dropout=0.25, num_classes=7):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, dropout=dropout)
