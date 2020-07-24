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


imgnet_quad_dir = '/scratch/nileshk/DeformParts/datasets/ImagenetQuads/train'
if osp.exists(imgnet_quad_dir):
    print('Streaming data from scratch')
    flags.DEFINE_string('imgnet_quad_dir', imgnet_quad_dir, 'Cub Data Directory')
else:
    flags.DEFINE_string('imgnet_quad_dir', '/home/nileshk/DeformParts/mask-rcnn.pytorch/data/imquad/train', 'Cub Data Directory')
flags.DEFINE_string('quad_class', 'rhino', 'quadruped class')

flags.DEFINE_string('imgnet_quad_anno_path', osp.join(cache_path, 'imnet_quad'), 'Directory where annotations are saved')

imnet_class2sysnet_list = {'rhino' : ['n02391994'], 'giraffe': ['n02439033'], 'camel': ['n02437312'], 
                           'hippo': ['n02398521'],
                           'fox': ['n02119022', 'n02119789', 'n02120079', 'n02120505',],
                           'bear': ['n02132136', 'n02133161'],
                           'leopard': ['n02128385'], 
                           'bison': ['n02410509'],
                           'buffalo':['n02408429', 'n02410702'],
                           'donkey': ['n02390640', 'n02390738'],
                           'goat': ['n02416519', 'n02417070'],
                           'german-sheperd': ['n02106662', 'n02107574', 'n02109047'],
                           'beest' : ['n02421449', 'n02422106'],
                           'kangaroo':['n01877812'],
                           'german-shepherd': ['n02106662', 'n02107574', 'n02109047'],
                           'beest': ['n02421449', 'n02422106'],
                           'pig': ['n02396427', 'n02395406', 'n02397096'],
                           'lion': ['n02129165', ],
                           'llama': ['n02437616', 'n02437971', ],
                           'tapir':  ['n02393580', 'n02393940' ],
                           'tiger': ['n02129604'],
                           'warthog': ['n02397096'],
                           'wolf': ['n02114367', 'n02114548', 'n02114712'],
                            }

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class ImgnetQuad(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''
    def __init__(self, opts,):
        super(ImgnetQuad, self).__init__(opts,)
        opts.dl_out_imnet = True
        opts.dl_out_pascal = False
        synset_ids = imnet_class2sysnet_list[opts.quad_class]
        all_annos = []
        self.imgnet_quad_anno_path = opts.imgnet_quad_anno_path
        for synset_id in synset_ids:
            anno_path = osp.join(self.imgnet_quad_anno_path, '{}_{}.mat'.format(synset_id, opts.split))
            anno = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']
            all_annos.extend(anno)


        self.pascal_img_dir = opts.imgnet_quad_dir
        self.imgnet_quad_dir = opts.imgnet_quad_dir
        # self.anno_sfm_path = osp.join(self.cub_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.dataset_source = []
        self.imnet_img_dir = self.imgnet_quad_dir
        
        self.anno = all_annos
        self.dataset_source.extend(['imgnet' for _ in range(len(self.anno))])
        
       
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15, 16]) - 1
        self.kp_names = ['L_Eye', 'R_Eye', 'Nose', 'L_EarBase', 'R_EarBase', 'Throat', 'Withers',
                         'TailBase', 'L_F_Elbow', 'R_F_Elbow', 'L_B_Elbow', 'R_B_Elbow',
                         'L_F_Paw', 'R_F_Paw', 'L_B_Paw', 'R_B_Paw']
        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.flip = opts.flip
        
        return

def quad_imgnet_dataloader(opts, shuffle=True):
    dset = ImgnetQuad(opts)
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
