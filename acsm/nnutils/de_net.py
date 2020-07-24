"""
Universal Correspondence net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import pdb
from . import net_blocks as nb
from . import unet
from ..utils import cub_parse
import torch.nn.functional as F
from torch.autograd import Variable, Function
flags.DEFINE_boolean('use_unet', False, 'Uses the unet version of the architecture')
flags.DEFINE_integer('output_img_size', 64, 'output img size after convolution')


class VGG16Conv(nn.Module):

    def __init__(self, n_blocks=4, use_pretrained=True):
        super(VGG16Conv, self).__init__()
        self.vgg16 = vgg16 = torchvision.models.vgg16(pretrained=use_pretrained)
        self.block2layerid = [2, 4, 7, 9, 12, 16]
        self.net = nn.Sequential(*list(vgg16.features.children())[0:self.block2layerid[n_blocks]])

    def forward(self, x):
        x = self.net.forward(x)
        return x


class DENetSimple(nn.Module):

    def __init__(self, opts):
        super(DENetSimple, self).__init__()
        modules = []
        modules.append(conv2d(3, 20, 5))
        modules.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(torch.nn.AvgPool2d(kernel_size=2, stride=2))
        modules.append(conv2d(20, 48, 5,))
        modules.append(conv2d(48, 64, 3,))
        modules.append(conv2d(64, 80, 3,))
        modules.append(conv2d(80, 256, 3,))
        modules.append(conv2d(256, 3, 1))
        self.encoder = nn.Sequential(*modules)
        
    def forward(self, feat):
        img = feat['img']
        Z = self.encoder.forward(img)
        return Z


def conv2d(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )


class DENetComplex(nn.Module):
    def __init__(self, opts):
        super(DENetComplex, self).__init__()
        self.unet_gen = unet.UnetConcatGenerator(input_nc=3, output_nc=3, num_downs=5,)
        self.ds_grid = cub_parse.get_sample_grid((opts.output_img_size, opts.output_img_size)).repeat(1, 1, 1, 1).cuda()
    def forward(self, feat):
        img = feat['img']
        Z = self.unet_gen.forward(img)
        bsize = len(img)
        Z = F.grid_sample(Z, self.ds_grid.repeat(bsize, 1,1,1,))
        return Z