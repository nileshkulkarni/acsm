from __future__ import absolute_import, division, print_function

'''
nice -n 20 python -m icn.experiments.cub.csp --n_data_workers=8 --batch_size=8 --reproject_loss_wt=1 --display_visuals --display_freq=20 --plot_scalars=True --name=birds_gt_camera_thresh0pt25_pred_uv2_3d --num_epochs=500 --debug_tb=True --save_visuals --save_visual_freq=2000 --use_html=True --save_epoch_freq=20 --use_normalized_uv=True --cycle_loss_wt=10.0 --cam_compute_ls=False --cycle_mask_loss_wt=10.0 --cycle_mask_min=0.25 --cmr_mean_shape=True --pred_xy_cycle=True --uv_to_3d_pred=True
'''

import multiprocessing
import os
import os.path as osp
import pdb
import threading
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pymesh
import scipy.io as sio
import scipy.misc

import torch
import torchvision
import torch.nn.functional as F
from absl import app, flags
from torch.autograd import Variable
from ...utils import mesh
from ...utils import metrics
from ...utils import cub_parse
from ...data import cub_de as cub_data
from ...data import pascal_imnet_de as pascal_imnet_data
from ...nnutils import de_loss_utils as loss_utils
# from ...utils import image as image_utils
from ...nnutils import geom_utils, de_net, train_utils
from ...nnutils import net_blocks as nb
from ...nnutils.nmr import NeuralRenderer
from ...utils import (bird_vis, cub_parse, mesh, render_utils, transformations,
                      visdom_render, visutil, gen_tps)

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('cachedir', cache_path, 'Cachedir')
flags.DEFINE_string('rendering_dir', osp.join('cache_path', 'rendering'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_string('result_dir', osp.join('cache_path', 'results'),
                    'Directory where intermittent renderings are saved')
flags.DEFINE_string('dataset', 'pascal', 'cub or globe or p3d')
flags.DEFINE_integer('seed', 0, 'seed for randomness')

cm = plt.get_cmap('jet')


class CSPTrainer(train_utils.Trainer):

    def define_model(self, ):
        opts = self.opts
        if opts.use_unet:
            self.model = de_net.DENetComplex(opts)
        else:
            self.model = de_net.DENetSimple(opts)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred',
                              self.opts.num_pretrain_epochs)

        self.downsample_grid = cub_parse.get_sample_grid(
            (opts.output_img_size, opts.output_img_size)).repeat(1, 1, 1, 1).to(self.device)
        self.grid = cub_parse.get_sample_grid((opts.img_size, opts.img_size)).repeat(1, 1, 1, 1).to(self.device)
        self.model.to(self.device)
        self.init_render()
        return

    def init_render(self, ):
        opts = self.opts
        kp_names = self.dataloader.dataset.kp_names
        nkps = len(kp_names)
        self.keypoint_cmap = [cm(i * 255//nkps) for i in range(nkps)]
        return

    def init_dataset(self, ):
        opts = self.opts
        
        if opts.dataset == 'cub':
            dataloader_fn = cub_data.cub_dataloader
        elif opts.dataset == 'pascal':
            dataloader_fn = pascal_imnet_data.pascal_dataloader
        else:
            raise 'Incorrect Dataset'

        self.dataloader = dataloader_fn(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return

    def append_bindex(kp_index):
        bIndex = torch.LongTensor(list(range(0, len(kp_index))))
        bIndex = bIndex[:, None, None]
        bIndex = bIndex.expand(kp_index[:, :, None, 0].shape).type(kp_index.type())
        kp_index = torch.cat([bIndex, kp_index], dim=-1)
        return kp_index

    def getRandomTPS(self, bsize):
        opts = self.opts
        g = []
        g_adj = []
        indices = torch.LongTensor(bsize * 2).random_(0, opts.ntps).to(self.device)
        g = torch.stack([self.tps['wfx'][indices], self.tps['wfy'][indices]], dim=1)
        downsample_grid = self.downsample_grid.repeat(bsize, 1, 1, 1)
        g_adj = F.grid_sample(g, downsample_grid) * (opts.output_img_size / opts.img_size)
        return g, g_adj

    def create_grid_from_flow(self, grid, flow, img_size, bSize):
        grid = grid.repeat(bSize, 1, 1, 1)
        new_grid = grid + flow.permute(0, 2, 3, 1) * 2 / img_size
        return new_grid

    def apply_warp(self, img, flow):
        bSize = len(img)
        img_size = img.shape[-1]
        new_grid = self.create_grid_from_flow(self.grid, flow, img_size, bSize)
        warp_img = F.grid_sample(img, new_grid)
        return warp_img, new_grid

    def set_input(self, batch):
        opts = self.opts
        codes_gt = {}
        input_imgs = batch['img'].type(self.Tensor)
        input_imgs_u = batch['img_u'].type(self.Tensor)
        input_imgs_v = batch['img_v'].type(self.Tensor)

        mask = batch['mask'].type(self.Tensor)
        mask_u = batch['mask_u'].type(self.Tensor)
        mask_v = batch['mask_v'].type(self.Tensor)

        bSize = len(input_imgs)
        for b in range(bSize):
            input_imgs_u[b] = self.resnet_transform(input_imgs_u[b])
            input_imgs_v[b] = self.resnet_transform(input_imgs_v[b])
            input_imgs[b] = self.resnet_transform(input_imgs[b])

        codes_gt['img'] = input_imgs
        codes_gt['img_u'] = input_imgs_u
        codes_gt['img_v'] = input_imgs_v

        self.inds = [k.item() for k in batch['inds']]
        mask = mask.to(self.device)
        mask_u = mask_u.to(self.device)
        mask_v = mask_v.to(self.device)

        codes_gt['mask'] = mask
        codes_gt['mask_u'] = mask_u
        codes_gt['mask_v'] = mask_v

        codes_gt['guv'] =  batch['guv'].type(self.Tensor).squeeze()
        codes_gt['downsample_grid'] = self.downsample_grid.repeat(bSize, 1, 1, 1)
        # codes_gt['guv_adj'] = F.grid_sample((codes_gt['guv'].permute(0,3,1,2).float() * opts.output_img_size)/opts.img_size,  codes_gt['downsample_grid'])
        
        ## Don't need to scale down by ouput_size/input_size since this is normalized transformation. guv \in (-1,1)
        codes_gt['guv_adj'] = F.grid_sample(codes_gt['guv'].permute(0,3,1,2).float(),  codes_gt['downsample_grid'])
        codes_gt['guv_adj'] = codes_gt['guv_adj'].permute(0,2,3,1).contiguous()

        codes_gt['kp_ind_u'] = batch['kp_ind_u'].type(self.Tensor)
        codes_gt['kp_ind_v'] = batch['kp_ind_v'].type(self.Tensor)
        codes_gt['kp_u'] = batch['kp_u'].type(self.Tensor)
        codes_gt['kp_v'] = batch['kp_v'].type(self.Tensor)
        codes_gt['kp'] = batch['kp'].type(self.Tensor)
        codes_gt['kp_vis'] = codes_gt['kp'][..., 2] > 0

        codes_gt['ds_mask_u'] = F.grid_sample(codes_gt['mask_u'],codes_gt['downsample_grid'])
        codes_gt['ds_mask_v'] = F.grid_sample(codes_gt['mask_v'],codes_gt['downsample_grid'])
        self.codes_gt = codes_gt
        return

    def define_criterion(self):
        opts = self.opts
        self.smoothed_factor_losses = {
            'nll': 0.0,
            'exp_dist': 0.0,
        }

        return

    def forward(self, ):
        opts = self.opts
        feed_dict = {}
        codes_gt = self.codes_gt
        feed_dict['img'] = torch.cat([codes_gt['img_u'], codes_gt['img_v']])
        bSize = len(codes_gt['img'])
        codes = {}
        img_feat = self.model.forward(feed_dict)

        codes['feat_u'] = img_feat[0:bSize]
        codes['feat_v'] = img_feat[bSize:]

        self.total_loss, self.loss_factors = loss_utils.code_loss(codes, codes_gt, opts)

        for k in self.smoothed_factor_losses.keys():
            if k in self.loss_factors.keys():
                self.smoothed_factor_losses[k] = 0.99 * \
                    self.smoothed_factor_losses[k] + 0.01 * self.loss_factors[k].item()

        codes['pred_u2v_map'] = self.loss_factors['pred_u2v_map']

        self.loss_factors.pop('pred_u2v_map')
        self.codes_pred = codes
        return

    def get_current_visuals(self, ):
        visuals = self.visuals_to_save(self.total_steps, count=1)[0]
        visuals.pop('ind')
        return visuals

    def transfer_points_via_cost_map(self,  v_img, u2v_map,):
        u2v_map = u2v_map.unsqueeze(0) # 1 x H x W x 2
        B, _, H, W = v_img.shape
        u2v_map = 2 * u2v_map / H - 1
        u_img = F.grid_sample(v_img, u2v_map)
        return u_img

    def visuals_to_save(self, total_steps, count=None):
        opts = self.opts
        batch_visuals = []
        codes_gt = self.codes_gt
        codes_pred = self.codes_pred
        if count == None:
            count = opts.save_visual_count

        vis_count = min(count,len(codes_gt['img']))
        for bx in range(vis_count):
            visuals = {}
            visuals['ind'] = "{:04}".format(self.inds[bx])
            visuals['img_u'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(codes_gt['img_u'].data[bx, None, :, :, :]))
            ds_img_u = F.grid_sample(codes_gt['img_u'].data[bx, None, :, :, :], self.downsample_grid)
            ds_img_v = F.grid_sample(codes_gt['img_v'].data[bx, None, :, :, :], self.downsample_grid)

            tfs_img_pred = self.transfer_points_via_cost_map(
                ds_img_v, codes_pred['pred_u2v_map'][bx].float())
            gt_u2v  = 0.5*(codes_gt['guv_adj'][bx] + 1)*opts.output_img_size
            tfs_img_gt = self.transfer_points_via_cost_map(ds_img_v, gt_u2v)


            visuals['pred_tfs_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(tfs_img_pred))
            visuals['gt_tfs_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(tfs_img_gt))
            visuals['ds_img_u'] = visutil.tensor2im(visutil.undo_resnet_preprocess(ds_img_u))
            visuals['ds_img_v'] = visutil.tensor2im(visutil.undo_resnet_preprocess(ds_img_v))
            
            visuals['img_kp_u'] = bird_vis.draw_keypoint_on_image(
                visuals['img_u'], self.codes_gt['kp_ind_u'][bx],
                self.codes_gt['kp_vis'][bx], self.keypoint_cmap)
            
            visuals['img_v'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(codes_gt['img_v'].data[bx, None, :, :, :]))
            
            visuals['img_kp_v'] = bird_vis.draw_keypoint_on_image(
                visuals['img_v'], self.codes_gt['kp_ind_v'][bx],
                self.codes_gt['kp_vis'][bx], self.keypoint_cmap)
            
            visuals.pop('img_u')
            visuals.pop('img_v')
            batch_visuals.append(visuals)
        return batch_visuals

    def get_current_points(self, ):
        pts_dict = {}
        return pts_dict

    def get_current_scalars(self, ):
        loss_dict = {
            'total_loss': self.smoothed_total_loss,
            'iter_frac': self.real_iter * 1.0 / self.total_steps
        }
        for k in self.smoothed_factor_losses.keys():
            loss_dict['loss_' + k] = self.smoothed_factor_losses[k]
        return loss_dict


FLAGS = flags.FLAGS


def main(_):

    seed = FLAGS.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.img_height = FLAGS.img_size
    FLAGS.img_width = FLAGS.img_size
    FLAGS.cache_dir = cache_path
    FLAGS.rendering_dir = osp.join(FLAGS.cache_dir, 'rendering', FLAGS.name)
    FLAGS.result_dir = osp.join(FLAGS.cache_dir, 'result', FLAGS.name)
    trainer = CSPTrainer(FLAGS)
    trainer.init_training()
    trainer.train()
    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
