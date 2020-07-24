"""
Data loader for pascal VOC categories.
Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
import scipy.io as sio
from absl import flags, app

import socket
import torch
from torch.utils.data.dataloader import default_collate
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..utils import transformations

from . import base as base_data
# -------------- flags ------------- #
# ---------------------------------- #
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

imnet_class2sysnet = {'horse' : 'n02381460', 'zebra': 'n02391049' , 'bear':'n02131653', 'sheep': 'n10588074', 'cow': 'n01887787',
                     'dog': 'n02381460',  }

# flags.DEFINE_string('pascal_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1', 'PASCAL Data Directory')
if 'vl-fb' in socket.getfqdn():
    flags.DEFINE_string('cub_dir', '/home/nileshk/data/DeformParts/CUB_200_2011/', 'CUB Data Directory')
elif 'umich' in socket.getfqdn():
    flags.DEFINE_string('cub_dir', '/data/nileshk/DeformParts/cub', 'CUB Data Directory')
else:
    cub_dir_flags = '/scratch/nileshk/DeformParts/datasets/cubs/'
    if osp.exists(cub_dir_flags):
        flags.DEFINE_string('cub_dir', cub_dir_flags, 'CUb Data Directory')
    else:
        flags.DEFINE_string('cub_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/cubs', 'CUb Data Directory')

flags.DEFINE_string('cub_anno_path', osp.join(cache_path, 'cub'), 'Directory where pascal annotations are saved')
flags.DEFINE_boolean('maskrcnn_anno', False, 'Use mask rcnn anno, ')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class CubDataset(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''
    def __init__(self, opts,):
        super(CubDataset, self).__init__(opts,)
        opts.dl_out_imnet = False
        opts.dl_out_pascal = True
        self.cub_img_dir = osp.join(opts.cub_dir, 'images')
        self.cub_cache_dir = opts.cub_anno_path
        if opts.maskrcnn_anno:
            self.anno_path = osp.join(self.cub_cache_dir, 'mrcnn', '%s_cub_cleaned.mat' % opts.split)
        else:
            self.anno_path = osp.join(self.cub_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        # self.anno_sfm_path = osp.join(self.cub_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.dataset_source = []
        self.imnet_img_dir = None
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.dataset_source.extend(['pascal' for _ in range(len(self.anno))])
        self.pascal_img_dir = self.cub_img_dir
        # self.anno_sfm = sio.loadmat(
        #     self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                         'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        opts.num_kps = len(self.kp_perm)


        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.flip = opts.flip

        return

def cub_dataloader(opts, shuffle=True):
    dset = CubDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class CubTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key

        sdset = CubDataset(opts)
        count = opts.number_pairs
        all_indices = [i for i in range(len(sdset))]
        rng = np.random.RandomState(len(sdset))
        pairs = zip(rng.choice(all_indices, count), rng.choice(all_indices, count))
        self.sdset = sdset
        self.tuples = pairs
        self.kp_names = sdset.kp_names
        self.kp_perm = sdset.kp_perm

    def __len__(self,):
        return len(self.tuples)

    def __getitem__(self, index):
        i1, i2 = self.tuples[index]
        b1 = self.sdset[i1]
        b2 = self.sdset[i2]
        if self.filter_key==1:
            return b1
        else:
            return b2


def cub_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = CubTestDataset(opts, filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)
