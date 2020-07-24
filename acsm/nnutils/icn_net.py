"""
Instance Correspondence net model.
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
from . import net_blocks as nb
from . import camera as cb  ## camera blocks
from . import unet
from . import resunet
from . import misc as misc_utils
from . import uv_to_3d
from .nmr import NeuralRenderer
from . import geom_utils
from . import loss_utils
import itertools
import math
import pdb
#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')
flags.DEFINE_boolean('render_mask', True, 'Render mask')
flags.DEFINE_boolean('render_depth', True, 'Render depth')
flags.DEFINE_boolean(
    'render_uv', False, 'Render uv to add loss for uv prediction'
)
flags.DEFINE_boolean('pred_mask', True, 'Learn to predict segmentation mask')
flags.DEFINE_boolean(
    'resnet_style_decoder', False, 'Use resnet style decoder for uvs'
)

flags.DEFINE_integer('resnet_blocks', 3, 'Encodes using resnet to layer')
flags.DEFINE_integer(
    'remove_skips', -1, 'Removes skip starting after which layer of UNet'
)
flags.DEFINE_integer(
    'warmup_pose_iter', 0, 'Warmup the pose predictor for multi '
)
flags.DEFINE_integer('warmup_deform_iter', 0, 'Warmup the part transform')
flags.DEFINE_integer('warmup_semi_supv', 0, 'Warmup the pose predictor')

flags.DEFINE_float(
    'reproject_loss_wt', 1, 'Reprojection loss bw uv and  loss weight.'
)
flags.DEFINE_float('kp_loss_wt', 1, 'Reprojection loss for keypoints')
flags.DEFINE_float('mask_loss_wt', 1.0, 'Mask loss wt.')
flags.DEFINE_float('cov_mask_loss_wt', 1.0, 'Mask loss wt.')
flags.DEFINE_float('con_mask_loss_wt', 1.0, 'Mask loss wt.')
flags.DEFINE_float('seg_mask_loss_wt', 1.0, 'Predicted Seg Mask loss wt.')
flags.DEFINE_float('depth_loss_wt', 1.0, 'Depth loss wt.')
flags.DEFINE_float('ent_loss_wt', 0.05, 'Entropy loss wt.')
flags.DEFINE_float('rot_reg_loss_wt', 0.1, 'Rotation Reg loss wt.')
flags.DEFINE_float('trans_reg_loss_wt', 10, 'Transform Reg loss wt.')
flags.DEFINE_float('delta_quat_reg_wt', 1.0, 'Regualizer delta quat')
flags.DEFINE_float('quat_reg_wt', 0.0, 'Regualizer delta quat')
flags.DEFINE_float('nmr_uv_loss_wt', 1.0, 'nmr uv loss wt')


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """
    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = nb.ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(
            batch_norm, 512, 256, stride=2, kernel_size=4
        )
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)
        self.resnet_feat = resnet_feat
        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat


class PartMembership(nn.Module):
    '''
    init_val n_verts x nparts
    '''
    def __init__(self, init_val):
        super(PartMembership, self).__init__()
        self.init_val = init_val

        self.membership = nn.Parameter(self.init_val * 0)
        self.membership.data.normal_(0, 0.0001)
        return

    def forward(self, ):
        # membership = torch.clamp(torch.tanh(0.0*self.membership) + self.init_val, 0,1)
        if False:
            membership = torch.clamp(
                torch.tanh(0.1 * self.membership) + self.init_val, 0, 1
            )
            membership = membership / (1E-12 + membership.sum(1)[..., None])
        if True:
            membership = self.init_val

        return membership


class ICPNet(nn.Module):
    def __init__(self, opts, init_stuff):
        super(ICPNet, self).__init__()
        self.opts = opts
        self.nc_encoder = 256
        self.uv_pred_dim = 3

        self.uv_sampler = init_stuff['uv_sampler']
        part_init = {
            'active_parts': init_stuff['active_parts'],
            'part_axis': init_stuff['part_axis']
        }
        self.kp_perm = init_stuff['kp_perm']
        self.part_perm = init_stuff['part_perm']
        if opts.resnet_style_decoder:
            self.unet_gen = resunet.ResNetConcatGenerator(
                input_nc=3,
                output_nc=self.uv_pred_dim + 1,
                n_blocks=opts.resnet_blocks
            )
        else:
            self.unet_gen = unet.UnetConcatGenerator(
                input_nc=3,
                output_nc=self.uv_pred_dim + 1,
                num_downs=5,
                remove_skips=opts.remove_skips
            )
            self.unet_innermost = self.unet_gen.get_inner_most()

        self.img_encoder = Encoder(
            (opts.img_size, opts.img_size), nz_feat=opts.nz_feat
        )

        if opts.num_hypo_cams == 1:
            self.cam_predictor = cb.SingleCamPredictor(
                nz_feat=opts.nz_feat,
                scale_lr_decay=opts.scale_lr_decay,
                scale_bias=opts.scale_bias,
                no_trans=opts.no_trans,
                part_init=part_init,
            )

        elif opts.multiple_cam:
            self.cam_predictor = cb.MultiCamPredictor(
                nz_feat=opts.nz_feat,
                num_cams=opts.num_hypo_cams,
                scale_lr_decay=opts.scale_lr_decay,
                scale_bias=opts.scale_bias,
                no_trans=opts.no_trans,
                part_init=part_init,
                euler_range=[
                    opts.az_euler_range, opts.el_euler_range,
                    opts.cyc_euler_range
                ]
            )

        self.mean_shape = init_stuff['mean_shape']

        self.init_nmrs()
        self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)
        self.cam_location = init_stuff['cam_location']
        self.offset_z = init_stuff['offset_z']
        self.kp_vertex_ids = init_stuff['kp_vertex_ids']

        self.part_membership = PartMembership(init_stuff['alpha'])
        img_size = (int(opts.img_size * 1.0), int(opts.img_size * 1.0))
        self.grid = misc_utils.get_img_grid(img_size).repeat(1, 1, 1, 1)
        return

    def init_nmrs(self, ):
        opts = self.opts
        tex_size = opts.tex_size
        devices = [0]
        if torch.cuda.device_count() > 1:
            devices = list(range(torch.cuda.device_count()))

        renderers = []
        for device in devices:
            multi_renderer_mask = nn.ModuleList(
                [
                    NeuralRenderer(opts.img_size, device=device)
                    for _ in range(opts.num_hypo_cams)
                ]
            )
            multi_renderer_depth = nn.ModuleList(
                [
                    NeuralRenderer(opts.img_size, device=device)
                    for _ in range(opts.num_hypo_cams)
                ]
            )
            renderers.append(
                {
                    'mask': multi_renderer_mask,
                    'depth': multi_renderer_depth,
                }
            )
            # renderers.append((multi_renderer_mask, multi_renderer_depth))

        self.renderers = renderers
        uv_sampler = self.uv_sampler
        texture_image = misc_utils.get_img_grid((1024, 1024)) * 0.5 + 0.5

        texture_image = torch.cat(
            [
                texture_image[:, :, None, 0], texture_image[:, :, None, 0] * 0 +
                1, texture_image[:, :, None, 1]
            ],
            dim=-1
        )

        texture_image = texture_image.permute(2, 0, 1).cuda(0)

        sampled_texture = torch.nn.functional.grid_sample(
            texture_image.unsqueeze(0), uv_sampler
        )
        sampled_texture = sampled_texture.squeeze().permute(1, 2, 0)
        sampled_texture = sampled_texture.view(
            sampled_texture.size(0), tex_size, tex_size, 3
        )

        sampled_texture = sampled_texture.unsqueeze(3).repeat(
            1, 1, 1, tex_size, 1
        )
        self.sampled_uv_texture_image = sampled_texture
        return

    def flip_inputs(self, inputs):
        flip_img = geom_utils.flip_image(inputs['img'])
        inputs['img'] = torch.cat([inputs['img'], flip_img])
        flip_mask = geom_utils.flip_image(inputs['mask'])
        inputs['mask'] = torch.cat([inputs['mask'], flip_mask])
        flip_mask_df = geom_utils.flip_image(inputs['mask_df'])
        inputs['mask_df'] = torch.cat([inputs['mask_df'], flip_mask_df])

        kp_perm = self.kp_perm.to(inputs['img'].get_device())

        new_kp = inputs['kp'].clone()
        new_kp[:, :, 0] = -1 * new_kp[:, :, 0]
        new_kp = new_kp[:, kp_perm, :]

        inputs['kp'] = torch.cat([inputs['kp'], new_kp])

        new_kp_valid = inputs['kp_valid'].clone()
        inputs['kp_valid'] = torch.cat([inputs['kp_valid'], new_kp_valid])
        inputs['inds'] = torch.cat([inputs['inds'], inputs['inds'] + 10000])

        inputs['contour'] = torch.cat(
            [inputs['contour'], inputs['flip_contour']]
        )
        return inputs

    def flip_predictions(self, codes_pred, true_size):

        # keys_to_copy = ['cam_probs', 'cam_sample_inds']
        device = codes_pred['cam_probs'].get_device()
        opts = self.opts
        if opts.multiple_cam:
            keys_to_copy = [
                'cam_probs',
            ]
            for key in keys_to_copy:
                codes_pred[key] = torch.cat(
                    [codes_pred[key][:true_size], codes_pred[key][:true_size]]
                )

        part_perm = self.part_perm.to(device)
        if opts.multiple_cam:
            keys_to_copy = ['part_transforms']
            for key in keys_to_copy:
                mirror_transforms_swaps = codes_pred[key][:true_size
                                                          ][:, :, part_perm, :]
                codes_pred[key] = torch.cat(
                    [codes_pred[key][:true_size], mirror_transforms_swaps]
                )

        ## mirror rotation
        camera = codes_pred['cam'][:true_size]
        if opts.multiple_cam:
            new_cam = cb.reflect_cam_pose(camera[:true_size])
            codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        else:
            new_cam = cb.reflect_cam_pose(camera[:true_size,
                                                 None, :]).squeeze(1)
            codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        return codes_pred

    def predict(self, img, deform):
        opts = self.opts
        predictions = {}
        device_id = img.get_device()
        unet_output = self.unet_gen.forward(img)
        img_feat = self.img_encoder.forward(img)
        uv_map = unet_output[:, 0:self.uv_pred_dim, :, :]

        mask = torch.sigmoid(unet_output[:, self.uv_pred_dim:, :, :])
        predictions['seg_mask'] = mask

        uv_map = torch.tanh(uv_map) * (1 - 1E-6)
        uv_map = torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
        uv_map_3d = uv_map.permute(0, 2, 3, 1).contiguous()
        uv_map = geom_utils.convert_3d_to_uv_coordinates(
            uv_map.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        uv_map = uv_map.permute(0, 2, 3, 1).contiguous()
        camera, part_transforms = self.cam_predictor.forward(img_feat)

        if opts.num_hypo_cams == 1:
            camera = camera.unsqueeze(1)
            part_transforms = part_transforms.unsqueeze(1)
            # delta_quat_transforms = delta_quat_transforms.unsqueeze(1)
            cam_probs = camera[:, :, 0:1] * 0 + 1
        elif opts.multiple_cam:
            camera, cam_probs = camera[:, :, :7], camera[:, :, 7:]

        membership = self.part_membership.forward().unsqueeze(0).repeat(
            len(uv_map), 1, 1
        )
        self.membership = membership.to(device_id)
        predictions['cam'] = camera
        predictions['cam_probs'] = cam_probs
        predictions['part_transforms'] = part_transforms
        predictions['uv_map'] = uv_map
        predictions['membership'] = membership.to(device_id)
        predictions['iter'] = 100000
        predictions = self.post_process_predictions(predictions, deform=deform)
        return predictions

    def forward(
        self,
        img,
        mask,
        mask_df,
        kp,
        kp_valid,
        inds,
        real_iter,
        contour,
        flip_contour=None,
        deform=False
    ):
        opts = self.opts
        inputs = {}
        predictions = {}
        device_id = img.get_device()
        inputs['img'] = img
        inputs['iter'] = real_iter
        inputs['mask'] = mask
        inputs['mask_df'] = mask_df
        inputs['kp'] = kp
        inputs['kp_valid'] = kp_valid
        inputs['inds'] = inds
        inputs['contour'] = contour

        if opts.flip_train:
            inputs['flip_contour'] = flip_contour
            inputs = self.flip_inputs(inputs)

        inputs['kp_vis'] = inputs['kp'][..., 2] > 0
        inputs['kps_vis'] = inputs['kp_vis'] * inputs['kp_valid']

        self.inputs = inputs
        img = inputs['img']

        predictions = self.predict(img, deform=deform)
        # predictions = self.post_process_predictions(predictions)
        inputs['xy_map'] = torch.cat(
            [self.grid[0:1, :, :, None, 0], self.grid[0:1, :, :, None, 1]],
            dim=-1
        ).unsqueeze(1)
        inputs['xy_map'] = inputs['xy_map'].to(device_id)
        return predictions, inputs

    '''
    This is called per camera hypothesis. 
    '''

    def _compute_geometric_predictions(
        self,
        verts,
        uv_map,
        camera,
        part_transforms,
        mask_renderer,
        depth_renderer,
    ):
        bsize = len(uv_map)
        img_size = uv_map.shape[1:3]
        device = camera.get_device()
        if part_transforms is not None:
            parts_rc = self.mean_shape['parts_rc']
            parts_rc = (torch.stack(parts_rc) * 1).to(device=device)
            parts_rc = parts_rc.unsqueeze(0).repeat(bsize, 1, 1)
            verts = geom_utils.apply_part_transforms(
                verts, self.mean_shape['parts'], parts_rc, part_transforms,
                self.membership
            )
        else:
            verts = verts[None, ].repeat(bsize, 1, 1)

        faces = (self.mean_shape['faces'] * 1).to(device)
        faces = faces[None, ...].repeat(bsize, 1, 1)
        mask_pred = mask_renderer.forward(verts, faces, camera)
        depth_pred = depth_renderer.forward(
            verts, faces, camera, depth_only=True
        )

        points3d = geom_utils.project_uv_to_3d(self.uv2points, verts, uv_map)
        project_points_cam_pred = geom_utils.project_3d_to_image(
            points3d, camera, self.offset_z
        )
        project_points_cam_z = (
            project_points_cam_pred[..., 2] - self.cam_location[2]
        )
        shape = (bsize, img_size[0], img_size[1])
        project_points_cam_z = project_points_cam_z.view(shape)
        shape = (bsize, img_size[0], img_size[1], 2)
        project_points = project_points_cam_pred[..., 0:2].view(shape)
        kp_verts = verts[:, self.kp_vertex_ids, :]
        kp_project = geom_utils.project_3d_to_image(
            kp_verts, camera, self.offset_z
        )
        kp_project = kp_project[...,
                                0:2].view(bsize, len(self.kp_vertex_ids), -1)
        verts_proj = geom_utils.project_3d_to_image(
            verts, camera, self.offset_z
        )[..., 0:2]

        predictions = {}
        predictions['verts'] = verts
        predictions['points3d'] = points3d
        predictions['project_points_cam_pred'] = project_points_cam_pred
        predictions['project_points_cam_z'] = project_points_cam_z
        predictions['project_points'] = project_points
        predictions['kp_project'] = kp_project
        predictions['verts_proj'] = verts_proj
        predictions['mask_render'] = mask_pred
        predictions['depth'] = depth_pred
        return predictions

    def post_process_predictions(self, predictions, deform):
        device_index = predictions['uv_map'].get_device()
        opts = self.opts
        b_size = len(predictions['uv_map'])
        if opts.flip_train:
            true_size = b_size // 2
            predictions = self.flip_predictions(
                predictions, true_size=true_size
            )

        real_iter = predictions['iter']

        verts = (self.mean_shape['verts'] * 1).to(device=device_index)
        geom_preds = []

        for cx in range(opts.num_hypo_cams):
            camera = predictions['cam'][:, cx]
            if deform:
                part_transforms = predictions['part_transforms'][:, cx]
            else:
                part_transforms = None

            renderers = self.renderers[0]
            multi_renderer_mask, multi_renderer_depth = renderers[
                'mask'], renderers['depth']
            geom_pred = self._compute_geometric_predictions(
                verts=verts,
                uv_map=predictions['uv_map'],
                camera=camera,
                part_transforms=part_transforms,
                mask_renderer=multi_renderer_mask[cx],
                depth_renderer=multi_renderer_depth[cx]
            )
            geom_preds.append(geom_pred)

        self.membership = predictions['membership']
        for key in geom_preds[0].keys():
            predictions[key] = torch.stack(
                [geom_preds[cx][key] for cx in range(opts.num_hypo_cams)],
                dim=1
            )

        if opts.warmup_pose_iter > real_iter:
            predictions['cam_probs'] = (1.0 / opts.num_hypo_cams) * (
                torch.zeros(predictions['cam_probs'].shape).float() + 1
            ).to(device_index)

        return predictions

    def reproject_loss_l2(self, project_points, grid_points, mask, dim=1):
        bsize, img_h, img_w = mask.shape
        non_mask_points = mask.view(bsize, -1).mean(1)
        mask = mask.unsqueeze(-1)
        loss = (mask * project_points - mask * grid_points)
        loss = loss.pow(2).sum(-1).view(bsize, -1).mean(-1)
        loss = loss / (non_mask_points + 1E-10)
        return loss

    def reproject_kp_loss(self, kp_pred, kp_gt, kp_vis):
        loss = ((kp_pred - kp_gt)**2).sum(-1) * kp_vis
        loss = loss.mean(-1) / (kp_vis.mean(-1) + 1E-4)
        return loss

    def depth_loss_fn(self, depth_render, depth_pred, mask):
        loss = torch.nn.functional.relu(depth_pred - depth_render).pow(2) * mask
        # loss = torch.nn.functional.relu(depth_pred-(depth_render + 1E-2)) * mask
        shape = loss.shape
        loss = loss.view(shape[0], -1)
        loss = loss.mean(-1)
        return loss

    def rotation_reg(self, cameras):
        opts = self.opts
        device = cameras.get_device()
        NC2_perm = list(itertools.permutations(range(opts.num_hypo_cams), 2))
        NC2_perm = torch.LongTensor(zip(*NC2_perm)).to(device)

        if len(NC2_perm) > 0:
            quats = cameras[:, :, 3:7]
            quats_x = torch.gather(
                quats,
                dim=1,
                index=NC2_perm[0].view(1, -1, 1).repeat(len(quats), 1, 4)
            )
            quats_y = torch.gather(
                quats,
                dim=1,
                index=NC2_perm[1].view(1, -1, 1).repeat(len(quats), 1, 4)
            )
            inter_quats = geom_utils.hamilton_product(
                quats_x, geom_utils.quat_conj(quats_y)
            )
            quatAng = geom_utils.quat2ang(inter_quats).view(
                len(inter_quats), opts.num_hypo_cams - 1, -1
            )
            quatAng = -1 * torch.nn.functional.max_pool1d(
                -1 * quatAng.permute(0, 2, 1), opts.num_hypo_cams - 1, stride=1
            ).squeeze()
            rotation_reg = (np.pi - quatAng).mean()
        else:
            rotation_reg = torch.zeros(1).mean().to(device)

        return rotation_reg

    def _compute_geometrics_losses(
        self, reprojected_points, reprojected_verts, reprojected_points_z,
        rendered_depth, rendered_mask, contours, mask_dt, mask, kps_projected,
        kps_gt, weight_dict
    ):
        device = reprojected_points.get_device()
        bsize = len(reprojected_points)
        xy_grid_gt = torch.cat(
            [self.grid[0:1, :, :, None, 0], self.grid[0:1, :, :, None, 1]],
            dim=-1
        ).to(device)

        reproject_loss = self.reproject_loss_l2(
            reprojected_points, xy_grid_gt, mask.squeeze(1)
        )

        if kps_gt is not None:
            kp_loss = self.reproject_kp_loss(
                kps_projected, kps_gt[:, :, 0:2], kps_gt[:, :, 2].float()
            )

        actual_depth_at_pixels = torch.nn.functional.grid_sample(
            rendered_depth.unsqueeze(1), reprojected_points.detach()
        )
        depth_loss = self.depth_loss_fn(
            actual_depth_at_pixels, reprojected_points_z.unsqueeze(1), mask
        )

        min_verts = []
        for bx in range(bsize):
            with torch.no_grad():
                mask_cov_err = (
                    reprojected_verts[bx, :, None, :] - contours[bx, :, :]
                )**2
                mask_cov_err = mask_cov_err.sum(-1)
                _, min_indices = torch.topk(-1 * mask_cov_err, k=4, dim=0)
                # _, min_indices = torch.min(mask_cov_err, dim=0)
            min_verts.append(reprojected_verts[bx][min_indices])
        min_verts = torch.stack(min_verts, dim=0)

        mask_cov_err = (min_verts - contours[:, None, ...])**2
        mask_cov_err = mask_cov_err.sum(-1).view(bsize, -1).mean(
            1
        )  # Topk dim is reduced.

        mask_con_err = mask_dt * rendered_mask[:, None]
        mask_con_err = mask_con_err.view(bsize, -1).mean(1)

        losses = {}
        losses['cyc'] = reproject_loss * weight_dict['cyc']
        losses['kp'] = kp_loss * weight_dict['kp']
        losses['vis'] = depth_loss * weight_dict['vis']
        losses['cov'] = mask_cov_err * weight_dict['cov']
        losses['con'] = mask_con_err * weight_dict['con']
        return losses

    def compute_geometrics_losses(self, predictions, inputs, weight_dict):
        opts = self.opts

        probs = predictions['cam_probs'].squeeze(2)
        losses = {}
        for hx in range(opts.num_hypo_cams):
            loss_hx = self._compute_geometrics_losses(
                reprojected_points=predictions['project_points'][:, hx],
                reprojected_verts=predictions['verts_proj'][:, hx],
                reprojected_points_z=predictions['project_points_cam_z'][:, hx],
                rendered_depth=predictions['depth'][:, hx],
                rendered_mask=predictions['mask_render'][:, hx],
                kps_projected=predictions['kp_project'][:, hx],
                contours=inputs['contour'],
                mask_dt=inputs['mask_df'],
                mask=inputs['mask'],
                kps_gt=inputs['kp'],
                weight_dict=weight_dict,
            )
            for key in loss_hx.keys():
                if key in losses:
                    losses[key].append(loss_hx[key])
                else:
                    losses[key] = [loss_hx[key]]

        for key in losses.keys():
            losses[key] = torch.stack(losses[key], dim=1)

        for key in losses.keys():
            if 'con' == key:
                losses[key] = losses[key].mean()
            elif 'cov' == key:
                losses[key] = losses[key].mean()
            else:
                losses[key] = losses[key] * probs
                losses[key] = losses[key].mean()

        if opts.num_hypo_cams > 1:
            rotation_reg = self.rotation_reg(predictions['cam'])
            losses['rot_reg'] = rotation_reg * opts.rot_reg_loss_wt
            dist_entp = -1 * (-torch.log(probs + 1E-9) * probs).sum(1).mean()
            dist_entp = weight_dict['ent'] * dist_entp
            losses['ent'] = dist_entp
        seg_mask_loss = torch.nn.functional.binary_cross_entropy(
            predictions['seg_mask'], inputs['mask']
        )
        losses['seg'] = seg_mask_loss * weight_dict['seg']

        regularize_trans = (predictions['part_transforms'][..., 1:4])**2
        regularize_trans = regularize_trans.sum(-1)  # 3
        regularize_trans = regularize_trans.sum(-1)  # nparts
        regularize_trans = regularize_trans.mean()
        losses['trans_reg'] = regularize_trans * weight_dict['trans_reg']
        return losses
