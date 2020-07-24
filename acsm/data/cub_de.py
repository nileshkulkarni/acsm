from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import collections

import scipy.misc
import scipy.linalg
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags
import cPickle as pkl
import torch
import multiprocessing
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pdb
from datetime import datetime
import sys
from ..utils import cub_parse
import numpy as np
import pdb
import pymesh
import re
import matplotlib.pyplot as plt
import scipy.misc
from ..utils import render_utils
from ..utils import visutil
from ..utils import transformations, gen_tps, bird_vis
from ..nnutils import geom_utils
from . import base as base_data
import itertools
import socket

if 'vl-fb' in socket.getfqdn():
    flags.DEFINE_string('cub_dir', '/data/home/cnris/nileshk/datasets/CUB_200_2011', 'CUB Data Directory')
elif 'umich' in socket.getfqdn():
    flags.DEFINE_string('cub_dir', '/data/nileshk/DeformParts/cub', 'CUB Data Directory')
else:
    cub_dir_flags = '/scratch/nileshk/DeformParts/datasets/cubs/'
    if osp.exists(cub_dir_flags):
        flags.DEFINE_string('cub_dir', cub_dir_flags, 'CUb Data Directory')
    else:
        flags.DEFINE_string('cub_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/cubs', 'CUb Data Directory')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')
flags.DEFINE_integer('ntps', 20, 'Number of TPS to sample from')
cm = plt.get_cmap('jet')


class CubDatasetDE(base_data.BaseDataset):

    def __init__(self, opts):
        super(CubDatasetDE, self).__init__(opts,)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir
        self.opts = opts
        self.img_dir = osp.join(self.data_dir, 'images')
        self.imnet_img_dir = None
        self.pascal_img_dir = self.img_dir
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.img_size = opts.img_size
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.dataset_source = []
        self.dataset_source.extend(['pascal' for _ in range(len(self.anno))])

        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                         'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        self.mean_shape = sio.loadmat(osp.join(opts.cub_cache_dir, 'uv', 'mean_shape.mat'))
        # self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'], self.mean_shape[
        #                                            'verts'], self.mean_shape['sphere_verts'])
        self.flip = opts.flip
        # pdb.set_trace()
        self.load_tps_warps()

    def load_tps_warps(self,):
        opts = self.opts
        tps_file_path = osp.join(opts.cache_dir, 'tps', 'tps_{}.mat'.format(opts.ntps))
        if osp.exists(tps_file_path):
            tps = sio.loadmat(tps_file_path)
        else:
            flow = gen_tps.create_random_tps(img_size=opts.img_size, tpslen=opts.ntps, stdev=0.20)
            tps = {}
            tps['backward'] = flow['backward']
            tps['forward'] = flow['forward']
            sio.savemat(tps_file_path, tps)
            tps = sio.loadmat(tps_file_path)

       
        self.tps_forward = {'wfx': torch.from_numpy(tps['forward']['wfx'][0,0]),
                    'wfy': torch.from_numpy(tps['forward']['wfy'][0,0])
                    }
        self.tps_backward = {'wfx': torch.from_numpy(tps['backward']['wfx'][0,0]),
                    'wfy': torch.from_numpy(tps['backward']['wfy'][0,0])
                    }

        self.downsample_grid = cub_parse.get_sample_grid(
            (opts.output_img_size, opts.output_img_size)).repeat(1, 1, 1, 1)
        self.grid = cub_parse.get_sample_grid((opts.img_size, opts.img_size)).repeat(1, 1, 1, 1)
        return

    def create_grid_from_flow(self, grid, flow, img_size):
        new_grid = grid + flow.permute(0, 2, 3, 1) * 2 / img_size
        return new_grid

    def apply_warp(self, img, flow, transform=None):
        img_size = img.shape[-1]
        new_grid = self.create_grid_from_flow(self.grid, flow, img_size)
        if transform is not None:
            inv_transform = torch.inverse(transform)
            temp_grid = torch.cat([new_grid, 0 * new_grid[:, :, :, 0:1] + 1], dim=-1).view(-1, 3).permute(1, 0)
            new_grid = torch.matmul(inv_transform, temp_grid)[0:2, :].permute(1, 0).view(new_grid.shape)
        warp_img = F.grid_sample(img, new_grid)
        return warp_img, new_grid

    def tf_scale(self, s):
        return torch.diag(torch.FloatTensor([s[0], s[1], 1.0]))

    def tf_rotate(self, theta):
        rotm = torch.FloatTensor([[np.cos(theta), np.sin(theta), 0],
                                  [-np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
        return rotm

    def tf_translate(self, translate):
        tm = torch.zeros(3, 3)
        tm[0, 2] = translate[0]
        tm[1, 2] = translate[1]
        tm[2, 2] = 1.0
        return tm

    def sample_random_tf(self,):
        scale, rotation, translation = np.random.uniform(
            0.8, 1.1, (2)), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.1, 0.1, size=(2))
        As = self.tf_scale(scale)
        Ar = self.tf_rotate(rotation)
        At = self.tf_translate(translation)
        A = torch.matmul(As, Ar) + At
        return A

    '''
    kp_src are indices of keypoints between 0 and 1.
    
    '''

    def find_correspondence_with_src(self, kp_src, new_grid, img_size):
        kp_dists = (kp_src[:, None, :] - new_grid.view(-1, 2)[None, :, :]).norm(p=2, dim=-1)
        _, min_inds = torch.min(kp_dists, dim=-1)
        wp_kp_inds_y = min_inds // img_size
        wp_kp_inds_x = min_inds - (min_inds // img_size) * img_size
        kps_warped = torch.stack([wp_kp_inds_x, wp_kp_inds_y], dim=-1)
        return kps_warped

    def compute_composite_flow(self, img2_fwd, A2, img1_back, A1):
        img_size = img2_fwd.shape[-1]
        new_grid = self.create_grid_from_flow(self.grid, img1_back, img_size)

        temp_grid = torch.cat([new_grid, 0 * new_grid[:, :, :, 0:1] + 1], dim=-1).view(-1, 3).permute(1, 0)
        tfs = torch.matmul(A2, torch.inverse(A1))
        new_grid = torch.matmul(tfs, temp_grid)[0:2, :].permute(1, 0).view(new_grid.shape)
        
        new_grid = self.create_grid_from_flow(new_grid, img2_fwd, img_size)
        return new_grid

    def tps_warp(self, img, mask, kp):
        opts = self.opts
        _, H, W = img.shape
        tpsx1 = np.random.choice(opts.ntps)
        tpsx2 = np.random.choice(opts.ntps)

        tps1_back = torch.stack([self.tps_backward['wfx'][tpsx1], self.tps_backward['wfy'][tpsx1]])
        tps1_fwd = torch.stack([self.tps_forward['wfx'][tpsx1], self.tps_forward['wfy'][tpsx1]])
        tps2_back = torch.stack([self.tps_backward['wfx'][tpsx2], self.tps_backward['wfy'][tpsx2]])
        tps2_fwd = torch.stack([self.tps_forward['wfx'][tpsx2], self.tps_forward['wfy'][tpsx2]])


        A1 = self.sample_random_tf()
        A2 = self.sample_random_tf()
        

        ## This will act the u image
        img_tps_u, new_grid_u = self.apply_warp(img[None, :, :, :], tps1_back[None, :, :, :], A1)
        mask_tps_u, _ = self.apply_warp(mask[None, None, :, :], tps1_back[None, :, :, :], A1)

        ## This will act the v image
        img_tps_v, new_grid_v = self.apply_warp(img[None,:,:,:], tps2_back[None, :, :, :], A2)
        mask_tps_v, _ = self.apply_warp(mask[None, None, :, :], tps2_back[None, :, :, :], A2)


        # pdb.set_trace()
        # img_np = visutil.saveTensorAsImage(img, 'img.png')
        # img_tps_u_np = visutil.saveTensorAsImage(img_tps_u[0], 'img1.png')
        # img_tps_v_np = visutil.saveTensorAsImage(img_tps_v[0], 'img2.png')
        ## This tells me how pixels on u map to pixels on v

        guv_fwd = self.compute_composite_flow(tps2_fwd[None,:,:,:], A2, tps1_back[None,:,:,:], A1)
        guv_back= self.compute_composite_flow(tps1_fwd[None,:,:,:], A1, tps2_back[None,:,:,:], A2)
        

        kp_ind = torch.round((kp * 0.5 + 0.5) * opts.img_size).clamp(0, opts.img_size - 1).long()

        kp_ind_u = self.find_correspondence_with_src(kp[:,0:2], new_grid_u, opts.img_size)
        kp_ind_u = torch.cat([kp_ind_u, kp_ind[:, 2:3]], dim=1)
        kp_u = kp_ind_u.clone().float(); kp_u = (kp_u/opts.img_size -0.5)*2


        kp_ind_v = self.find_correspondence_with_src(kp[:,0:2], new_grid_v, opts.img_size)
        kp_ind_v = torch.cat([kp_ind_v, kp_ind[:, 2:3]], dim=1)
        kp_v = kp_ind_v.clone().float(); kp_v = (kp_v/opts.img_size -0.5)*2

        
        # imgU2V = F.grid_sample(img_tps_u, guv_back)
        # visutil.saveTensorAsImage(imgU2V[0], 'imguv.png')
        # imgV2U = F.grid_sample(img_tps_v, guv_fwd)
        # visutil.saveTensorAsImage(imgV2U[0], 'imgv2u.png')
        # pdb.set_trace()
        # keypoint_cmap = [cm(i * 17) for i in range(15)]
        # img_np_kp = bird_vis.draw_keypoint_on_image(img_np, kp_ind, kp_ind[:, 2] > 150, keypoint_cmap)
        # img_tps_u_np_kp = bird_vis.draw_keypoint_on_image(img_tps_u_np, kp_ind_u, kp_ind[:, 2] > 150, keypoint_cmap)
        # img_tps_v_np_kp = bird_vis.draw_keypoint_on_image(img_tps_v_np, kp_ind_v, kp_ind[:, 2] > 150, keypoint_cmap)

        # visutil.save_image(img_np_kp, 'img_kp.png')
        # visutil.save_image(img_tps_u_np_kp, 'img_kp1.png')
        # visutil.save_image(img_tps_v_np_kp, 'img_kp2.png')
        # pdb.set_trace()
        return {
                'img_v': img_tps_v[0], 'img_u': img_tps_u[0],
                'mask_v': mask_tps_v[0] , 'mask_u': mask_tps_u[0],
                'grid_v': new_grid_v[0], 'grid_u': new_grid_u[0]  ,
                'kp_v': kp_v, 'kp_u': kp_u, 'kp': kp,
                'kp_ind': kp_ind, 'kp_ind_v': kp_ind_v, 'kp_ind_u': kp_ind_u,
                'guv': guv_fwd[0],
                }

    def __getitem__(self, index):
        elem = super(CubDatasetDE, self).__getitem__(index)  # 'img' , 'kp', 'mask'
        elem['img'] = torch.from_numpy(elem['img']).float()
        elem['mask'] = torch.from_numpy(elem['mask']).float()
        elem['kp'] = torch.from_numpy(elem['kp']).float()
        warped_imgs = self.tps_warp(elem['img'],  elem['mask'], elem['kp'])
        elem.update(warped_imgs)
        return elem


def cub_dataloader(opts, shuffle=True):
    dset = CubDatasetDE(opts)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


