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
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..utils import transformations
from ..utils import cub_parse
from ..utils import render_utils
from ..utils import visutil
from ..utils import transformations, gen_tps, bird_vis
from ..nnutils import geom_utils

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
        self.load_tps_warps()
        return

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
        elem = super(PascalImNetDataset, self).__getitem__(index)  # 'img' , 'kp', 'mask'
        elem['img'] = torch.from_numpy(elem['img']).float()
        elem['mask'] = torch.from_numpy(elem['mask']).float()
        elem['kp'] = torch.from_numpy(elem['kp']).float()
        warped_imgs = self.tps_warp(elem['img'],  elem['mask'], elem['kp'])
        elem.update(warped_imgs)
        return elem



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
