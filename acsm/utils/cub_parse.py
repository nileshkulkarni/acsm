from __future__ import division
from __future__ import print_function
import copy
import csv
import json
import numpy as np
import scipy.linalg
import scipy.io as sio
import os
import os.path as osp
import cPickle as pickle
import cPickle as pkl
import torch
from torch.autograd import Variable
from . import transformations
import pdb
from collections import defaultdict
import math
import torch.nn as nn
from ..nnutils import geom_utils


def get_sample_grid(img_size):
    x = torch.linspace(-1, 1, img_size[1]).view(1, -1).repeat(img_size[0], 1)
    y = torch.linspace(-1, 1, img_size[0]).view(-1, 1).repeat(1, img_size[1])
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
    grid.unsqueeze(0)
    return grid


def collate_pair_batch(examples):
    batch = {}
    for key in examples[0]:
        if key in [
            'kp_valid', 'contour', 'mask_df', 'flip_contour', 'kp_uv', 'img',
            'inds', 'neg_inds', 'mask', 'kp', 'pos_inds', 'sfm_pose', 'anchor'
        ]:
            batch[key] = torch.cat(
                [examples[i][key] for i in range(len(examples))], dim=0
            )
    return batch


def normalize(point_3d):
    return point_3d / (1E-10 + np.linalg.norm(point_3d))


## Adds bindex as the first dimension.
'''
Input faces: B x N x 3

Output : B x N x 4 (bindex, v1, v2, v3)
'''


def add_bIndex(faces):
    bsize = len(faces)
    bIndex = torch.LongTensor([i for i in range(bsize)]).type(faces.type())
    bIndex = bIndex.view(-1, 1, 1).repeat(1, faces.shape[1], 1)
    faces = torch.cat([bIndex, faces], dim=-1)
    return faces


def load_mean_shape(mean_shape_path, device='cuda:0'):
    if type(mean_shape_path) == str:
        mean_shape = sio.loadmat(mean_shape_path)
    else:
        mean_shape = mean_shape_path
    # mean_shape['bary_cord'] = torch.from_numpy(mean_shape['bary_cord']).float().to(device)
    mean_shape['uv_map'] = torch.from_numpy(mean_shape['uv_map']
                                            ).float().to(device)
    mean_shape['uv_verts'] = torch.from_numpy(mean_shape['uv_verts']
                                              ).float().to(device)
    mean_shape['verts'] = torch.from_numpy(mean_shape['verts']
                                           ).float().to(device)
    mean_shape['sphere_verts'] = torch.from_numpy(mean_shape['sphere_verts']
                                                  ).float().to(device)
    mean_shape['face_inds'] = torch.from_numpy(mean_shape['face_inds']
                                               ).long().to(device)
    mean_shape['faces'] = torch.from_numpy(mean_shape['faces']
                                           ).long().to(device)
    if 'sublookup_type' in mean_shape.keys():
        mean_shape['sublookup_type'] = torch.from_numpy(
            mean_shape['sublookup_type']
        ).long().to(device)
        mean_shape['sublookup_offset'] = torch.from_numpy(
            mean_shape['sublookup_offset']
        ).float().to(device)
        mean_shape['sublookup_length'] = torch.from_numpy(
            mean_shape['sublookup_length']
        ).float().to(device)
        mean_shape['sublookup_faceinds'] = torch.from_numpy(
            mean_shape['sublookup_faceinds']
        ).long().to(device)

    return mean_shape