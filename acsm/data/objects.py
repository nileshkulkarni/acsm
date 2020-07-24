"""
Code borrowed from Code borrowed from https://github.com/shubhtuls/toe/blob/master/data/objects.py
Data loader for object classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
import socket
import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import base2 as base_data

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

if 'vl-fb' in socket.getfqdn():
    base_dir = '/home/nileshk/data/DeformParts/'

flags.DEFINE_string(
    'imgnet_dir', osp.join(base_dir, 'Imagenet/ImageSets'),
    'Imagenet images directory'
)
flags.DEFINE_string(
    'imgnet_anno_path', osp.join(cache_path, 'imnet_quad'),
    'Directory where annotations are saved'
)

flags.DEFINE_string(
    'voc_dir', osp.join(base_dir, 'VOCdevkit/VOC2012/JPEGImages'),
    'VOC images directory'
)
flags.DEFINE_string(
    'pascal_anno_path', osp.join(cache_path, 'pascal'),
    'Directory where annotations are saved'
)

flags.DEFINE_string(
    'p3d_dir', osp.join(base_dir, 'PASCAL3D/PASCAL3D+_release1.1'),
    'PASCAL Data Directory'
)
flags.DEFINE_string(
    'p3d_anno_path', osp.join(cache_path, 'p3d'),
    'Directory where p3d annotations are saved'
)

flags.DEFINE_string(
    'cub_dir', osp.join(base_dir, 'CUB_200_2011/'), 'CUB Data Directory'
)
flags.DEFINE_string(
    'cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory'
)

flags.DEFINE_string('category', 'bird', 'object category')
# ------------ Misc Utils ------------ #
# ------------------------------------ #


def standardize_annotation(anno):
    if isinstance(anno[0].parts, (np.ndarray, np.generic)):
        return anno
    else:
        synset = anno[0].rel_path.split('_')[0]
        for ix in range(len(anno)):
            anno[ix].rel_path = osp.join(synset, anno[ix].rel_path)
            anno[ix].parts = anno[ix].parts.T.T
        return anno


imnet_class2sysnet_list = {
    'rhino': ['n02391994'],
    'giraffe': ['n02439033'],
    'camel': ['n02437312'],
    'hippo': ['n02398521'],
    'fox': [
        'n02119022',
        'n02119789',
        'n02120079',
        'n02120505',
    ],
    'bear': ['n02132136', 'n02133161', 'n02131653'],
    'leopard': ['n02128385'],
    'bison': ['n02410509'],
    'buffalo': ['n02408429', 'n02410702'],
    'donkey': ['n02390640', 'n02390738'],
    'goat': ['n02416519', 'n02417070'],
    'beest': ['n02421449', 'n02422106'],
    'kangaroo': ['n01877812'],
    'german-shepherd': ['n02106662', 'n02107574', 'n02109047'],
    'pig': ['n02396427', 'n02395406', 'n02397096'],
    'lion': [
        'n02129165',
    ],
    'llama': ['n02437616', 'n02437971'],
    'tapir': ['n02393580', 'n02393940'],
    'tiger': ['n02129604'],
    'warthog': ['n02397096'],
    'wolf': ['n02114367', 'n02114548', 'n02114712'],
    'horse': ['n02381460'],
    'zebra': ['n02391049'],
    'sheep': ['n10588074'],
    'cow': ['n01887787'],
    'dog': ['n02381460'],
    'elephant': ['n02504013'],
}


# -------------- Datasets ------------ #
# ------------------------------------ #
class CUBDataset(base_data.BaseKpCamDataset):
    '''
    CUB Data loader
    '''
    def __init__(self, opts, filter_key=None, pascal_only=False):
        super(CUBDataset, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(
            self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split
        )
        self.anno_sfm_path = osp.join(
            self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split
        )

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            pdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True
        )['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True
        )['sfm_anno']

        self.num_imgs = len(self.anno)
        self.kp_names = [
            'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg',
            'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'
        ]
        self.kp_perm = np.array(
            [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]
        ) - 1


class ImgnetQuadDataset(base_data.BaseDataset):
    '''
    Quadruped Data loader
    '''
    def __init__(self, opts, filter_key=None, pascal_only=False):
        super(ImgnetQuadDataset, self).__init__(opts, filter_key=filter_key)
        data_cache_dir = opts.imgnet_anno_path
        data_cache_dir = osp.join(cache_path, 'imnet_quad_pointrend')
        synset_ids = imnet_class2sysnet_list[opts.category]
        self.img_dir = opts.imgnet_dir
        self.kp_perm = np.array([i for i in range(16)])
        self.kp_names = None
        all_annos = []
        self._out_kp = True
        for synset_id in synset_ids:
            anno_path = osp.join(
                data_cache_dir, '{}_{}.mat'.format(synset_id, opts.split)
            )
            if osp.exists(anno_path):
                anno = sio.loadmat(
                    anno_path, struct_as_record=False, squeeze_me=True
                )['images']
                anno = standardize_annotation(anno)
                all_annos.extend(anno)

        self.anno = all_annos
        self.num_imgs = len(self.anno)


class ImgnetPascalQuadDataset(base_data.BaseDataset):
    def __init__(self, opts, filter_key=None, pascal_only=False):
        super(ImgnetPascalQuadDataset,
              self).__init__(opts, filter_key=filter_key)
        self._out_kp = True

        ## Pascal Annotations
        pascal_cache_dir = opts.pascal_anno_path
        kp_path = osp.join(
            pascal_cache_dir, 'data', '{}_kps.mat'.format(opts.category)
        )
        pascal_anno_path = osp.join(
            pascal_cache_dir, 'data',
            '{}_{}.mat'.format(opts.category, opts.split)
        )
        pascal_anno = sio.loadmat(
            pascal_anno_path, struct_as_record=False, squeeze_me=True
        )['images']
        self.kp_perm = sio.loadmat(
            kp_path, struct_as_record=False, squeeze_me=True
        )['kp_perm_inds'] - 1
        self.kp_names = sio.loadmat(
            kp_path, struct_as_record=False, squeeze_me=True
        )['kp_names'].tolist()
        opts.num_kps = len(self.kp_perm)

        ## Imagenet Annotations
        data_cache_dir = opts.imgnet_anno_path
        synset_ids = imnet_class2sysnet_list[opts.category]
        imgnet_anno = []

        for synset_id in synset_ids:
            anno_path = osp.join(
                data_cache_dir, '{}_{}.mat'.format(synset_id, opts.split)
            )
            if osp.exists(anno_path):
                anno = sio.loadmat(
                    anno_path, struct_as_record=False, squeeze_me=True
                )['images']
                anno = standardize_annotation(anno)
                imgnet_anno.extend(anno)

        self.img_dir = '/'
        for ix in range(len(imgnet_anno)):
            imgnet_anno[ix].rel_path = osp.join(
                opts.imgnet_dir[1:], imgnet_anno[ix].rel_path
            )
            imgnet_anno[ix].parts = np.zeros((3, opts.num_kps))
        for ix in range(len(pascal_anno)):
            pascal_anno[ix].rel_path = osp.join(
                opts.voc_dir[1:], pascal_anno[ix].rel_path
            )

        self.anno = np.array([])
        self.anno = np.concatenate([pascal_anno])
        # self.anno = np.concatenate([imgnet_anno])
        if opts.split == 'train' and not pascal_only:
            self.anno = np.concatenate([self.anno, imgnet_anno])
        self.num_imgs = len(self.anno)


class P3dDataset(base_data.BaseKpCamDataset):
    ''' 
    P3d Data loader
    '''
    def __init__(self, opts, filter_key=None):
        super(P3dDataset, self).__init__(opts, filter_key=filter_key)
        self.img_dir = osp.join(opts.p3d_dir, 'Images')
        self.kp_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_kps.mat'.format(opts.category)
        )
        self.anno_path = osp.join(
            opts.p3d_anno_path, 'data',
            '{}_{}.mat'.format(opts.category, opts.split)
        )
        self.anno_sfm_path = osp.join(
            opts.p3d_anno_path, 'sfm',
            '{}_{}.mat'.format(opts.category, opts.split)
        )

        # Load the annotation file.
        self.cad_dir = osp.join(opts.p3d_dir, 'CAD', opts.category)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True
        )['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True
        )['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True
        )['kp_perm_inds'] - 1

        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)


#----------- Data Loader ----------#
#----------------------------------#
def cub_data_loader(opts, shuffle=True):
    return base_data.base_loader(
        CUBDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle
    )


def imnet_quad_data_loader(opts, shuffle=True):
    return base_data.base_loader(
        ImgnetQuadDataset,
        opts.batch_size,
        opts,
        filter_key=None,
        shuffle=shuffle
    )


def imnet_pascal_quad_data_loader(opts, pascal_only=False, shuffle=True):
    return base_data.base_loader(
        ImgnetPascalQuadDataset,
        opts.batch_size,
        opts,
        filter_key=None,
        shuffle=shuffle,
        pascal_only=pascal_only,
    )


def p3d_data_loader(opts, shuffle=True):
    return base_data.base_loader(
        P3dDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle
    )
