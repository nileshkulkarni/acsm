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

imnet_class2sysnet = {'horse': 'n02381460', 'zebra': 'n02391049', 'bear': 'n02131653',
                      'sheep': 'n10588074', 'cow': 'n01887787', 'dog': 'n02381460', 'elephant': 'n02504013'}

# flags.DEFINE_string('pascal_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/PASCAL3D+_release1.1', 'PASCAL Data Directory')


flags.DEFINE_string('imnet_anno_path', osp.join(
    cache_path, 'imnet'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('pascal_anno_path', osp.join(
    cache_path, 'pascal'), 'Directory where pascal annotations are saved')


opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class PascalImNetDataset(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''

    def __init__(self, opts,):
        super(PascalImNetDataset, self).__init__(opts,)
        self.pascal_img_dir = osp.join(opts.pascal_dir, 'JPEGImages')
        sysnetId = imnet_class2sysnet[opts.pascal_class]
        self.imnet_img_dir = osp.join(opts.imnet_dir, 'ImageSets', sysnetId)
        self.pascal_cache_dir = opts.pascal_anno_path
        self.imnet_cache_dir = opts.imnet_anno_path
        self.kp_path = osp.join(self.pascal_cache_dir,
                                'data', '{}_kps.mat'.format(opts.pascal_class))
        
        self.pascal_anno_path = osp.join(
            self.pascal_cache_dir, 'data', '{}_{}.mat'.format(opts.pascal_class, opts.split))
        self.pascal_anno_path = osp.join(
            self.pascal_cache_dir, 'data', '{}_{}.mat'.format(opts.pascal_class, opts.split))

        self.imnet_anno_path = osp.join(
            self.imnet_cache_dir, 'data', '{}_{}.mat'.format(sysnetId, opts.split))

        if opts.dl_out_pascal:
            self.anno_pascal = sio.loadmat(
                self.pascal_anno_path, struct_as_record=False, squeeze_me=True)['images']

        self.anno_imnet = sio.loadmat(
            self.imnet_anno_path, struct_as_record=False, squeeze_me=True)['images']

        self.anno = np.array([])
        self.dataset_source = []

        if opts.dl_out_pascal:
            self.anno = np.concatenate([self.anno, self.anno_pascal])
            self.dataset_source.extend(
                ['pascal' for _ in range(len(self.anno_pascal))])

        if opts.dl_out_imnet:
            self.anno = np.concatenate([self.anno, self.anno_imnet])
            self.dataset_source.extend(
                ['imnet' for _ in range(len(self.anno_imnet))])

        # pdb.set_trace()
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1
        self.kp_names = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_names'].tolist()
        opts.num_kps = len(self.kp_perm)

        # pdb.set_trace()
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.flip = opts.flip
        return


def pascal_dataloader(opts, shuffle=True):
    dset = PascalImNetDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


class PascalTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key

        sdset = PascalImNetDataset(opts)
        count = opts.number_pairs
        all_indices = [i for i in range(len(sdset))]
        rng = np.random.RandomState(len(sdset))
        pairs = zip(rng.choice(all_indices, count),
                    rng.choice(all_indices, count))
        # pairs = [(17, 47), (106, 20), (44,20),]
        # pairs = [(44,20),]
        self.sdset = sdset
        self.tuples = pairs
        self.kp_names = sdset.kp_names
        self.kp_perm = sdset.kp_perm

    def __len__(self,):
        return len(self.tuples)

    def __getitem__(self, index):
        i1, i2 = self.tuples[index]
        # i2 = 21
        # i1 = 57
        # i2 = 39
        # i1 = 96
        # i2 = 58
        # i1 = 62
        # i2 = 28
        # i1 = 11

        # Good example.
        # i1 = 106
        # i2 = 20

        # ##
        # i1 = 44
        # i2 = 20

        b1 = self.sdset[i1]
        b2 = self.sdset[i2]
        # pdb.set_trace()
        if self.filter_key == 1:
            return b1
        else:
            return b2


def pascal_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = PascalTestDataset(opts, filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)
