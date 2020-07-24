from __future__ import absolute_import, division, print_function
'''
nice -n 20 python -m icn.experiments.cub.csp --n_data_workers=8 --batch_size=8 --reproject_loss_wt=1 --display_visuals --display_freq=20 --plot_scalars=True --name=birds_gt_camera_thresh0pt25_pred_uv2_3d --num_epochs=500 --debug_tb=True --save_visuals --save_visual_freq=2000 --use_html=True --save_epoch_freq=20 --use_normalized_uv=True --cycle_loss_wt=10.0 --cam_compute_ls=False --cycle_mask_loss_wt=10.0 --cycle_mask_min=0.25 --cmr_mean_shape=True --pred_xy_cycle=True --uv_to_3d_pred=True
'''

import matplotlib
matplotlib.use('Agg')
import multiprocessing
import os
import os.path as osp
import pdb
import threading
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc
import cPickle as pkl
import torch
import torchvision
import copy
from absl import app, flags
from torch.autograd import Variable
from ...utils import mesh
from ...utils import metrics
from ...data import objects as objects_data
from ...nnutils import loss_utils
from ...nnutils import geom_utils, icn_net, train_utils, model_utils
from ...nnutils.nmr import NeuralRenderer
from ...nnutils import uv_to_3d
from ...utils import (
    bird_vis, cub_parse, mesh, render_utils, transformations, visdom_render,
    visutil, image
)

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
flags.DEFINE_string('cachedir', cache_path, 'Cachedir')
flags.DEFINE_string(
    'rendering_dir', osp.join('cache_path', 'rendering'),
    'Directory where intermittent renderings are saved'
)
flags.DEFINE_string(
    'result_dir', osp.join('cache_path', 'results'),
    'Directory where intermittent renderings are saved'
)
flags.DEFINE_string('dataset', 'globe', 'cub or globe or pascal')
flags.DEFINE_string('ft_name', '', 'start from a pretrained model')
flags.DEFINE_integer('ft_pretrain_epochs', -1, 'cub or globe or pascal')

flags.DEFINE_float('split_size', 1.0, 'Split size of the train set')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_boolean('semi_supv', False, 'Train in a semi supv setting,')

cm = plt.get_cmap('jet')


class CSPTrainer(train_utils.Trainer):
    def define_model(self, ):
        opts = self.opts
        self.img_size = opts.img_size
        self.offset_z = 5.0

        init_stuff = {
            'alpha': self.mean_shape['alpha'],
            'active_parts': self.part_active_state,
            'part_axis': self.part_axis_init,
            'kp_perm': self.kp_perm,
            'part_perm': self.part_perm,
            'mean_shape': self.mean_shape,
            'cam_location': self.cam_location,
            'offset_z': self.offset_z,
            'kp_vertex_ids': self.kp_vertex_ids,
            'uv_sampler': self.uv_sampler,
        }
        model = icn_net.ICPNet(opts, init_stuff)
        self.gpu_count = 1
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            self.gpu_count = int(torch.cuda.device_count())

        if opts.ft_pretrain_epochs > 0:
            is_dataparallel_model = self.dataparallel_model(
                'pred', self.opts.ft_pretrain_epochs, network_dir=opts.ft_name
            )

            if is_dataparallel_model and torch.cuda.device_count() == 1:
                model = torch.nn.DataParallel(model)

            self.load_network(
                model,
                'pred',
                opts.ft_pretrain_epochs,
                network_dir=opts.ft_name
            )

        if opts.num_pretrain_epochs > 0:
            is_dataparallel_model = self.dataparallel_model(
                'pred', self.opts.num_pretrain_epochs
            )
            if is_dataparallel_model and torch.cuda.device_count() == 1:
                model = torch.nn.DataParallel(model)

            self.load_network(model, 'pred', self.opts.num_pretrain_epochs)

        self.model = model

        self.model.to(self.device)

        self.upsample_img_size = (
            (opts.img_size // 64) * (2**6), (opts.img_size // 64) * (2**6)
        )

        self.grid = cub_parse.get_sample_grid(self.upsample_img_size
                                              ).repeat(1, 1, 1,
                                                       1).to(self.device)

        self.model.to(self.device)
        self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)
        # self.renderer.render_mean_bird_with_uv()
        return

    def init_membership(self, verts, parts):
        nverts = len(verts)
        nparts = len(parts)
        membership = torch.FloatTensor(nverts, nparts).zero_()
        for px in range(nparts):
            vert_ids = parts[px]
            membership[vert_ids, px] = 1.0
        membership = membership / membership.sum(1).view(-1, 1)
        return membership

    def init_render(self, ):
        opts = self.opts
        model_obj_dir = osp.join(self.save_dir, 'model')
        visutil.mkdir(model_obj_dir)
        self.model_obj_path = osp.join(
            model_obj_dir, 'mean_{}.obj'.format(opts.pascal_class)
        )

        nkps = len(self.kp_vertex_ids)
        self.keypoint_cmap = [cm(i * 255 // nkps) for i in range(nkps)]

        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size
        )
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(
            -1, len(faces_np), opts.tex_size * opts.tex_size, 2
        )

        self.verts_obj = self.mean_shape['verts']

        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        self.renderer = visdom_render.RendererWrapper(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap,
            self.opts
        )

        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        renderer_no_light = visdom_render.RendererWrapper(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap,
            self.opts
        )
        renderer_no_light.vis_rend.set_light_status(False)
        renderer_no_light.vis_rend.set_bgcolor((255, 255, 255))
        self.renderer_no_light = renderer_no_light

        self.sphere_uv_img = scipy.misc.imread(
            osp.join(opts.cachedir, 'color_maps', 'sphere.png')
        )
        self.sphere_uv_img = torch.FloatTensor(self.sphere_uv_img) / 255
        self.sphere_uv_img = self.sphere_uv_img.permute(2, 0, 1)
        return

    def init_dataset(self, ):
        opts = self.opts
        if opts.category == 'bird':
            self.pascal_dataloader = objects_data.cub_data_loader(opts, )
            self.all_dataloader = objects_data.cub_data_loader(opts, )
        elif opts.category in ['horse', 'sheep', 'cow']:
            self.pascal_dataloader = objects_data.imnet_pascal_quad_data_loader(
                opts, pascal_only=True
            )
            self.all_dataloader = objects_data.imnet_pascal_quad_data_loader(
                opts,
            )
        else:
            self.pascal_dataloader = objects_data.imnet_quad_data_loader(opts, )
            self.all_dataloader = objects_data.imnet_quad_data_loader(opts, )

        self.dataloader = self.all_dataloader
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if opts.warmup_semi_supv > 0:
            self.dataloader = self.pascal_dataloader

        self.kp_perm = self.pascal_dataloader.dataset.kp_perm
        if self.kp_perm is not None:
            self.kp_perm = torch.LongTensor(self.kp_perm)

        self.preload_model_data()
        self.init_render()
        return

    def preload_model_data(self, ):
        opts = self.opts
        model_dir, self.mean_shape, self.mean_shape_np = model_utils.load_template_shapes(
            opts, device_mapping=self.device
        )
        dpm, parts_data, self.kp_vertex_ids = model_utils.init_dpm(
            self.dataloader.dataset.kp_names, model_dir, self.mean_shape,
            opts.parts_file
        )
        opts.nparts = self.mean_shape['alpha'].shape[1]
        self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(
            model_dir, self.save_dir, dpm, parts_data, suffix=''
        )
        return

    def init_parts_point_of_rotation(self, verts, parts):
        nparts = len(parts)
        parts_cg = []
        for px in range(nparts):
            part_verts = verts[parts[px]]
            cg = part_verts.mean(0)
            parts_cg.append(cg)

        parts_cg = torch.stack(parts_cg)
        return parts_cg

    def set_input(self, batch):
        opts = self.opts
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = [k.item() for k in batch['inds']]
        self.input_img_tensor = input_imgs.to(self.device)
        mask = (mask > 0.5).float()
        self.mask = mask.to(self.device)
        self.mask_df = batch['mask_df'].type(self.Tensor).to(self.device)

        img_size = self.input_img_tensor.shape[-1]
        self.codes_gt = {}
        self.codes_gt['inds'] = torch.LongTensor(self.inds).to(self.device)
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)
        self.codes_gt['kp_valid'] = batch['kp'][:, :, 2].to(self.device) > 0.5

        self.codes_gt['contour'] = (batch['contour']).float().to(self.device)
        self.codes_gt['contour'
                      ] = (self.codes_gt['contour'] / img_size - 0.5) * 2
        if opts.flip_train:
            self.codes_gt['flip_contour'] = (batch['flip_contour']).float().to(
                self.device
            )
            self.codes_gt['flip_contour'] = (
                self.codes_gt['flip_contour'] / img_size - 0.5
            ) * 2

        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (self.codes_gt['kp'] * 0.5 + 0.5) * img_size
        self.codes_gt['kps_vis'] = kps_vis
        self.codes_gt['kps_ind'] = kps_ind
        self.codes_gt['kps_vis'] = kps_vis * self.codes_gt['kp_valid']
        return

    def define_criterion(self):
        opts = self.opts
        self.smoothed_factor_losses = self.sc_dict
        # self.smoothed_factor_losses = defaultdict(float)

    def forward(self, ):
        opts = self.opts
        feed_dict = {}
        codes_gt = self.codes_gt
        feed_dict['img'] = self.input_img_tensor
        feed_dict['mask'] = self.mask.unsqueeze(1)
        feed_dict['mask_df'] = self.mask_df.unsqueeze(1)
        feed_dict['iter'] = self.real_iter
        feed_dict['kp'] = codes_gt['kp']
        feed_dict['kp_valid'] = codes_gt['kp_valid']
        feed_dict['inds'] = codes_gt['inds']
        feed_dict['contour'] = codes_gt['contour']
        if opts.flip_train:
            feed_dict['flip_contour'] = codes_gt['flip_contour']
        else:
            feed_dict['flip_contour'] = None

        deform = self.real_iter > opts.warmup_deform_iter
        predictions, inputs = self.model.forward(
            img=feed_dict['img'],
            mask=feed_dict['mask'],
            mask_df=feed_dict['mask_df'],
            kp=feed_dict['kp'],
            kp_valid=feed_dict['kp_valid'],
            inds=feed_dict['inds'],
            real_iter=self.real_iter,
            contour=feed_dict['contour'],
            flip_contour=feed_dict['flip_contour'],
            deform=deform,
        )

        weight_dict = {}
        weight_dict['con'] = opts.con_mask_loss_wt
        weight_dict['cov'] = opts.cov_mask_loss_wt

        weight_dict['trans_reg'] = opts.trans_reg_loss_wt
        weight_dict['seg'] = opts.seg_mask_loss_wt
        weight_dict['kp'] = opts.kp_loss_wt

        if opts.warmup_pose_iter > self.real_iter:
            weight_dict['cyc'] = 0.0 * opts.reproject_loss_wt
            weight_dict['vis'] = 0.0 * opts.depth_loss_wt
            weight_dict['ent'] = 0.0 * opts.ent_loss_wt
        else:
            weight_dict['cyc'] = opts.reproject_loss_wt
            weight_dict['vis'] = opts.depth_loss_wt
            weight_dict['ent'] = opts.ent_loss_wt

        losses = self.model.compute_geometrics_losses(
            predictions, inputs, weight_dict
        )
        total_loss = 0
        for key, loss in losses.items():
            total_loss += loss

        self.total_loss = total_loss

        index = 0
        codes_pred = predictions

        for key in inputs.keys():
            codes_gt[key] = inputs[key]
        codes_gt['kps_ind'] = (codes_gt['kp'] * 0.5 + 0.5) * self.img_size

        self.loss_factors = losses

        for key, loss in losses.items():
            self.register_scalars({key: loss.item()})

        if opts.warmup_semi_supv < self.real_iter:
            if self.dataloader is not self.all_dataloader:
                print('Changed to all dataloader')
                self.dataloader = self.all_dataloader

        self.codes_pred = codes_pred
        self.codes_gt = codes_gt

        return

    def filter_as_per_mask(
        self,
        tensor,
        mask,
    ):
        return torch.masked_select(
            tensor,
            mask.squeeze().unsqueeze(-1) > 0,
        )

    def get_current_visuals(self, ):
        visuals = self.visuals_to_save(
            self.total_steps, count=1, for_visdom=True
        )[0]
        visuals.pop('ind')
        return visuals

    def visuals_to_save(self, total_steps, count=None, for_visdom=False):
        renderer = self.renderer
        opts = self.opts
        mean_shape_np = self.mean_shape_np

        if count is None:
            count = min(opts.save_visual_count, len(self.codes_gt['img']))

        batch_visuals = []
        inds = self.codes_gt['inds'].data.cpu().numpy()
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        uv_map = self.codes_pred['uv_map']
        camera = self.codes_pred['cam']
        kp_valid = self.codes_gt['kp_valid']
        bsize = len(mask)
        results_dir = osp.join(
            opts.result_dir, "{}".format(opts.split), "{}".format(total_steps)
        )

        visual_ids = list(range(count))
        if opts.flip_train and (not for_visdom):
            count = max(count, 2)
            bper_gpu = bsize // self.gpu_count
            offset = bper_gpu // 2
            visual_ids = []
            for gx in range(self.gpu_count):
                for bx in range(bper_gpu):
                    visual_ids.append(bx + gx * bper_gpu * 2)
                    visual_ids.append(bx + gx * bper_gpu * 2 + offset)

        # pdb.set_trace()
        visual_ids = visual_ids[0:count]
        for b in visual_ids:
            visuals = {}
            visuals['ind'] = "{:04}".format(inds[b])
            if opts.render_mask and opts.multiple_cam:
                mask_renders = self.codes_pred['mask_render'][
                    b, :, None, ...].repeat(1, 3, 1, 1).data.cpu()
                mask_renders = (mask_renders.numpy() *
                                255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['mask_render'] = visutil.image_montage(
                    mask_renders, nrow=min(3, opts.num_hypo_cams // 3 + 1)
                )

            if opts.render_depth and opts.multiple_cam:
                all_depth_hypo = (
                    self.codes_pred['mask_render'][b] *
                    self.codes_pred['depth'][b]
                )[:, None, :, :].repeat(1, 3, 1, 1).data.cpu() / 50.0
                all_depth_hypo = (all_depth_hypo.numpy() *
                                  255).astype(np.uint8).transpose(0, 2, 3, 1)
                visuals['all_depth'] = visutil.image_montage(
                    all_depth_hypo, nrow=min(3, opts.num_hypo_cams // 3 + 1)
                )

            if opts.multiple_cam:
                vis_cam_hypotheses = renderer.render_all_hypotheses(
                    camera[b],
                    probs=self.codes_pred['cam_probs'][b],
                    verts=self.codes_pred['verts'][b]
                )
                visuals.update(vis_cam_hypotheses)

            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(img.data[b, None, :, :, :])
            )
            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'], self.codes_gt['kps_ind'][b],
                self.codes_gt['kps_vis'][b], self.keypoint_cmap
            )
            visuals['img_kp_rp'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'],
                self.codes_pred['kp_project'][b][0] * 128 + 128,
                self.codes_gt['kps_vis'][b], self.keypoint_cmap
            )

            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :]
            )
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu()
            )
            # visuals['model'] = (self.render_model_using_cam(self.codes_pred['cam'][b])*255).astype(np.uint8)
            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b], self.codes_pred['project_points'][b][0]
            )

            contour_indx = (
                opts.img_size * (self.codes_gt['contour'][b] * 0.5 + 0.5)
            ).data.cpu().numpy().astype(np.int)
            visuals['contour'] = renderer.visualize_contour_img(
                contour_indx, opts.img_size
            )

            img_ix = (
                torch.FloatTensor(visuals['z_img']).permute(2, 0, 1)
            ) / 255
            visuals['a_overlay_uvmap'] = bird_vis.sample_UV_contour(
                img_ix,
                uv_map.float()[b].cpu(),
                self.sphere_uv_img,
                mask[b].float().cpu(),
                real_img=True
            )

            visuals['a_overlay_uvmap'] = visutil.tensor2im(
                [visuals['a_overlay_uvmap'].data]
            )

            max_ind = torch.argmax(self.codes_pred['cam_probs'][b].squeeze()
                                   ).item()
            renderer.update_verts(self.codes_pred['verts'][b][max_ind])
            if not opts.multiple_cam:
                camera = camera.unsqueeze(1)
            uv_overlayed_model_imgs = []

            for nx in range(opts.num_hypo_cams):
                self.renderer_no_light.update_verts(
                    self.codes_pred['verts'][b][nx]
                )
                tex_bird_img, _ = self.renderer_no_light.wrap_texture(
                    self.sphere_uv_img.cuda(),
                    camera[b][nx],
                    True,
                    tex_size=opts.tex_size
                )
                uv_overlayed_model_imgs.append(tex_bird_img)

            visuals['model_uv_colored'] = visutil.image_montage(
                uv_overlayed_model_imgs,
                nrow=min(3, opts.num_hypo_cams // 3 + 1)
            )
            texture_vps = renderer.render_model_using_nmr(
                uv_map.data[b],
                img.data[b],
                mask.data[b],
                camera[b][0],
                upsample_texture=True,
                other_vps=True
            )
            visuals.update(texture_vps)

            if opts.pred_mask:
                visuals['pred_mask'] = visutil.tensor2im(
                    self.codes_pred['seg_mask'].data.repeat(1, 3, 1,
                                                            1)[b, None, :, :, :]
                )

            batch_visuals.append(visuals)
            if not for_visdom:
                mean_shape = {
                    'verts':
                    self.codes_pred['verts'][b][max_ind].data.cpu().numpy(),
                    'faces': mean_shape_np['faces'],
                    'uv_verts': mean_shape_np['uv_verts']
                }
                bird_vis.save_obj_with_texture(
                    '{:04}'.format(inds[b]), results_dir,
                    visuals['texture_img'], mean_shape
                )

        return batch_visuals

    def get_current_points(self, ):
        pts_dict = {}
        return pts_dict

    def get_current_scalars(self, ):
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
            break

        loss_dict = {
            'total_loss': self.smoothed_total_loss,
            'iter_frac': self.real_iter * 1.0 / self.total_steps,
            'learning_rate': learning_rate
        }
        # loss_dict['grad_norm'] = self.grad_norm
        for k in self.sc_dict.keys():
            # if np.abs(self.smoothed_factor_losses[k]) > 1E-5 or 'loss_{}'.format(k) in loss_dict.keys():
            loss_dict['loss_' + k] = self.sc_dict[k]
        return loss_dict


FLAGS = flags.FLAGS


def main(_):

    seed = FLAGS.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    FLAGS.cache_dir = cache_path
    FLAGS.rendering_dir = osp.join(FLAGS.cache_dir, 'rendering', FLAGS.name)
    FLAGS.result_dir = osp.join(FLAGS.cache_dir, 'result', FLAGS.name)
    trainer = CSPTrainer(FLAGS)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)
