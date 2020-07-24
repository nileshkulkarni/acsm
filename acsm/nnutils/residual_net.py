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
import chamfer

from . import net_blocks as nb
from . import camera as cb  # camera blocks
from . import unet
from ..utils import cub_parse
from .nmr import NeuralRenderer
from . import geom_utils
from . import loss_utils
from . import chamfer_ext as chamfer_dist
import scipy.ndimage
import math
import pdb
#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')
flags.DEFINE_integer('nz_UV_height', 256 // 3, 'image height')
flags.DEFINE_integer('nz_UV_width', 256 // 3, 'image width')
flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')
flags.DEFINE_boolean('render_mask', True, 'Render mask')
flags.DEFINE_boolean('render_depth', True, 'Render depth')
flags.DEFINE_boolean('pred_mask', True, 'Learn to predict segmentation mask')
flags.DEFINE_integer('warmup_pose_iter', 0,
                     'Warmup the pose predictor for multi ')
flags.DEFINE_integer('warmup_deform_iter', 0, 'Warmup the part transform')
flags.DEFINE_integer('warmup_semi_supv', 0, 'Warmup the pose predictor')


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
            batch_norm, 512, 256, stride=2, kernel_size=4)
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

        self.membership = nn.Parameter(self.init_val*0)
        self.membership.data.normal_(0, 0.0001)
        return

    def forward(self, ):
        # membership = torch.clamp(torch.tanh(0.0*self.membership) + self.init_val, 0,1)
        if False:
            membership = torch.clamp(torch.tanh(
                0.1 * self.membership) + self.init_val, 0, 1)
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

        part_init = {'active_parts': init_stuff['active_parts'],
                     'part_axis': init_stuff['part_axis']}
        self.kp_perm = init_stuff['kp_perm']
        self.part_perm = init_stuff['part_perm']
        self.uv_sampler = init_stuff['uv_sampler']

        self.unet_gen = unet.UnetConcatGenerator(
            input_nc=3, output_nc=self.uv_pred_dim + 1, num_downs=5,)
        self.unet_innermost = self.unet_gen.get_inner_most()
        self.img_encoder = Encoder(
            (opts.img_size, opts.img_size), nz_feat=opts.nz_feat)

        if opts.num_hypo_cams == 1:
            self.cam_predictor = cb.SingleCamPredictor(nz_feat=opts.nz_feat,
                                                       aze_ele_quat=False,
                                                       scale_lr_decay=opts.scale_lr_decay,
                                                       scale_bias=opts.scale_bias,
                                                       no_trans=opts.no_trans,
                                                       single_axis_pred=opts.single_axis_pred,
                                                       delta_quat_pred=opts.delta_quat_pred,
                                                       part_init=part_init, pascal_class=opts.pascal_class)

        elif opts.multiple_cam:
            self.cam_predictor = cb.MultiCamPredictor(nz_feat=opts.nz_feat,
                                                      num_cams=opts.num_hypo_cams,
                                                      aze_ele_quat=opts.az_ele_quat,
                                                      scale_lr_decay=opts.scale_lr_decay,
                                                      scale_bias=opts.scale_bias,
                                                      no_trans=opts.no_trans,
                                                      single_axis_pred=opts.single_axis_pred,
                                                      delta_quat_pred=opts.delta_quat_pred,
                                                      part_init=part_init,
                                                      pascal_class=opts.pascal_class,)

        self.mean_shape = init_stuff['mean_shape']

        self.init_nmrs()
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        self.cam_location = init_stuff['cam_location']
        self.offset_z = init_stuff['offset_z']
        self.kp_vertex_ids = init_stuff['kp_vertex_ids']

        self.part_membership = PartMembership(init_stuff['alpha'])
        # self.part_predictor = PartPredictor(nz_feat = opts.nz_feat, init_val=init_stuff['alpha'])
        small_img_size = ((opts.img_size // 64) * (2**5),
                          (opts.img_size // 64) * (2**5))
        img_size = (int(opts.img_size * 1.0), int(opts.img_size * 1.0))
        self.grid = cub_parse.get_sample_grid(img_size).repeat(
            1, 1, 1, 1)
        self.sdf = BatchedSignedDistance(opts.batch_size)
        return

    def init_nmrs(self, ):
        opts = self.opts
        tex_size = opts.tex_size
        devices = [0]
        if torch.cuda.device_count() > 1:
            devices = list(range(torch.cuda.device_count()))

        renderers = []
        for device in devices:
            multi_renderer_mask = nn.ModuleList([NeuralRenderer(
                opts.img_size, device=device) for _ in range(opts.num_hypo_cams)])
            multi_renderer_depth = nn.ModuleList([NeuralRenderer(
                opts.img_size, device=device) for _ in range(opts.num_hypo_cams)])
            multi_uv_renderer = nn.ModuleList([NeuralRenderer(
                opts.img_size, device=device) for _ in range(opts.num_hypo_cams)])
            for renderer in multi_uv_renderer:
                renderer.set_light_status(False)
            renderers.append(
                {'mask': multi_renderer_mask, 'depth': multi_renderer_depth,
                 'uv': multi_uv_renderer})

        self.renderers = renderers

        uv_sampler = self.uv_sampler
        texture_image = cub_parse.get_sample_grid((1024, 1024)) * 0.5 + 0.5
        texture_image = torch.cat([texture_image[:, :, None, 0],
                                   texture_image[:, :, None, 0]*0 + 1,
                                   texture_image[:, :, None, 1]], dim=-1)
        texture_image = texture_image.permute(2, 0, 1).cuda(0)

        sampled_texture = torch.nn.functional.grid_sample(
            texture_image.unsqueeze(0), uv_sampler)
        sampled_texture = sampled_texture.squeeze().permute(1, 2, 0)
        sampled_texture = sampled_texture.view(
            sampled_texture.size(0), tex_size, tex_size, 3)

        sampled_texture = sampled_texture.unsqueeze(
            3).repeat(1, 1, 1, tex_size, 1)
        self.sampled_uv_texture_image = sampled_texture

    def flip_inputs(self, inputs):
        flip_img = geom_utils.flip_image(inputs['img'])
        inputs['img'] = torch.cat([inputs['img'], flip_img])
        flip_mask = geom_utils.flip_image(inputs['mask'])
        inputs['mask'] = torch.cat([inputs['mask'], flip_mask])
        flip_mask_df = geom_utils.flip_image(inputs['mask_df'])
        inputs['mask_df'] = torch.cat([inputs['mask_df'], flip_mask_df])

        kp_perm = self.kp_perm.to(inputs['img'].get_device())

        new_kp = inputs['kp'].clone()
        new_kp[:, :, 0] = -1*new_kp[:, :, 0]
        new_kp = new_kp[:, kp_perm, :]

        inputs['kp'] = torch.cat([inputs['kp'], new_kp])

        new_kp_valid = inputs['kp_valid'].clone()
        inputs['kp_valid'] = torch.cat([inputs['kp_valid'], new_kp_valid])
        inputs['inds'] = torch.cat([inputs['inds'], inputs['inds']+10000])

        inputs['contour'] = torch.cat(
            [inputs['contour'], inputs['flip_contour']])
        return inputs

    def flip_predictions(self, codes_pred, true_size):

        # keys_to_copy = ['cam_probs', 'cam_sample_inds']
        device = codes_pred['cam_probs'].get_device()
        opts = self.opts
        if opts.multiple_cam:
            keys_to_copy = ['cam_probs', ]
            for key in keys_to_copy:
                codes_pred[key] = torch.cat(
                    [codes_pred[key][:true_size], codes_pred[key][:true_size]])

        part_perm = self.part_perm.to(device)
        if opts.multiple_cam:
            keys_to_copy = ['part_transforms', 'delta_part_transforms']
            for key in keys_to_copy:
                mirror_transforms_swaps = codes_pred[key][:true_size][:,
                                                                      :, part_perm, :]
                codes_pred[key] = torch.cat(
                    [codes_pred[key][:true_size], mirror_transforms_swaps])

        # mirror rotation
        camera = codes_pred['cam'][:true_size]
        if opts.multiple_cam:
            new_cam = cb.reflect_cam_pose(camera[:true_size])
            codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        else:
            new_cam = cb.reflect_cam_pose(
                camera[:true_size, None, :]).squeeze(1)
            codes_pred['cam'] = torch.cat([camera[:true_size], new_cam])
        return codes_pred

    def forward_camera(self, img,):
        img_feat = self.img_encoder.forward(img)
        camera, part_transforms, delta_quat_transforms = self.cam_predictor.forward(
            img_feat)
        return camera, part_transforms, delta_quat_transforms

    def forward_verts(self, camera, part_transforms, membership):
        opts = self.opts
        real_iter = self.real_iter
        bsize = len(camera)

        device_index = camera.get_device()
        verts = (self.mean_shape['verts']*1).to(device=device_index)
        if opts.warmup_deform_iter > real_iter:
            n_verts = verts.shape[0]
            verts = verts[None, None, ...].expand(
                (bsize, opts.num_hypo_cams, n_verts, 3))
        else:
            parts_rc = self.mean_shape['parts_rc']
            parts_rc = (torch.stack(parts_rc)*1).to(device=device_index)
            parts_rc = parts_rc.unsqueeze(0).repeat(bsize, 1, 1)

            verts = []

            for cx in range(opts.num_hypo_cams):
                part_transforms_cx = part_transforms[:, cx]
                verts_cx = geom_utils.apply_part_transforms(verts,
                                                            self.mean_shape['parts'],
                                                            parts_rc,
                                                            part_transforms_cx,
                                                            membership)
                verts.append(verts_cx)
            verts = torch.stack(verts, dim=1)
        return verts

    def forward_uv_rendering(self, faces, verts, camera):
        opts = self.opts
        bsize = len(faces)
        device = verts.get_device()
        multi_uv_renderers = self.renderers[device]['uv']
        uv_renders = []
        uv_texture = self.sampled_uv_texture_image.to(device)
        uv_texture = uv_texture.unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        for cx in range(opts.num_hypo_cams):
            uv_render = multi_uv_renderers[cx].forward(
                verts[:, cx], faces, camera[:, cx], uv_texture)
            uv_renders.append(uv_render)
        uv_renders = torch.stack(uv_renders, dim=1)
        return uv_renders

    def forward_apporximate_uv_image(self, masks, uv_renders,):
        '''
            Compute the approixmate uv_mapping by doing a nearest neigbhour
            search over the rendered uv. This is a CPU implementation.
            Uses scipy.ndimage

            Proposed algo:
            Compute non-zero from masks, and uv_renders.
            computer champer indices using the
        '''
        distChamfer = chamfer_dist.chamferDist()
        approximate_uvs = []
        device = uv_renders.get_device()
        with torch.no_grad():
            bsize, num_cams, _, imgH, imgW, = uv_renders.shape
            rendered_masks = uv_renders[..., 1, :, :] > 0.5
            for bx, (rendered_mask, mask) in enumerate(zip(rendered_masks, masks)):
                mask_nonzero_inds = torch.nonzero(mask[0])
                approx_uv_batch= []
                pdb.set_trace()
                bx_df_indices= sdf.forward(rendered_mask.data.cpu())
                pdb.set_trace()
                sample_indices = df_indices / imgH
                sample_indices = sample_indices * 2 - 1
                sample_indices= sample_indices.cuda(device)
                sampled_uv= torch.nn.functional.grid_sample(
                        uv_renders[bx, cx, None, ], sample_indices)

                pdb.set_trace()
                # for cx in range(num_cams):
                #     render_mask_nonzero_inds = torch.nonzero(rendered_mask[cx])
                #     _, df_indices = scipy.ndimage.distance_transform_edt(
                #         rendered_mask[cx].data.cpu().numpy(), return_indices=True)
                #     sample_indices = torch.FloatTensor(df_indices) / imgH
                #     sample_indices = sample_indices * 2 - 1
                #     pdb.set_trace()
                #     sample_indices = sample_indices.cuda(device)
                #     sampled_uv = torch.nn.functional.grid_sample(
                #         uv_renders[bx, cx, None, ], sample_indices)
                #     approx_uv_batch.append(sampled_uv)

                #     distmask, distrender, indsmask2render, indsrender2mask = distChamfer(
                #         mask_nonzero_inds.float().unsqueeze(0),
                #         render_mask_nonzero_inds.float().unsqueeze(0))
                #     pdb.set_trace()

        return

    def forward(self, img, mask, mask_df, kp, kp_valid, inds, real_iter,
                contour, flip_contour=None):
        opts= self.opts

        predictions= {}
        inputs= {}
        device_id= img.get_device()
        inputs['img']= img
        inputs['iter']= real_iter
        inputs['mask']= mask
        inputs['mask_df']= mask_df
        inputs['kp']= kp
        inputs['kp_valid']= kp_valid
        inputs['inds']= inds
        inputs['contour']= contour
        inputs['flip_contour']= flip_contour
        self.real_iter= real_iter

        if opts.flip_train:
            inputs= self.flip_inputs(inputs)

        inputs['kp_vis']= inputs['kp'][..., 2] > 0
        inputs['kps_vis']= inputs['kp_vis'] * inputs['kp_valid']

        self.inputs= inputs
        img= inputs['img']
        camera, part_transforms, delta_quat_transforms= self.forward_camera(
            img)

        if opts.num_hypo_cams == 1:
            camera= camera.unsqueeze(1)
            part_transforms= part_transforms.unsqueeze(1)
            delta_quat_transforms= delta_quat_transforms.unsqueeze(1)
            cam_probs= camera[:, :, 0:1]*0 + 1
        elif opts.multiple_cam:
            camera, cam_probs= camera[:, :, :7], camera[:, :, 7:]

        if opts.flip_train:
            predictions['cam']= camera
            predictions['part_transforms']= part_transforms
            predictions['delta_part_transforms']= delta_quat_transforms
            predictions['cam_probs']= cam_probs
            predictions= self.flip_predictions(predictions, len(img)//2)
            camera, part_transforms, delta_quat_transforms, cam_probs= predictions[
                'cam'], predictions['part_transforms'], predictions[
                'delta_part_transforms'], predictions['cam_probs']

        membership= self.mean_shape['alpha'].to(
            device_id).unsqueeze(0).repeat(len(img), 1, 1)
        verts= self.forward_verts(camera, part_transforms, membership)
        faces= self.mean_shape['faces']
        faces= faces[None, ...].repeat(len(img), 1, 1)
        uv_rendering= self.forward_uv_rendering(faces, verts, camera)

        naive_uvs= self.forward_apporximate_uv_image(
            inputs['mask'], uv_rendering)

        pdb.set_trace()
        unet_output= self.unet_gen.forward(img)

        uv_map= unet_output[:, 0:self.uv_pred_dim, :, :]

        mask= torch.sigmoid(unet_output[:, self.uv_pred_dim:, :, :])
        predictions['seg_mask']= mask

        uv_map= torch.tanh(uv_map) * (1 - 1E-6)
        uv_map= torch.nn.functional.normalize(uv_map, dim=1, eps=1E-6)
        uv_map_3d= uv_map.permute(0, 2, 3, 1).contiguous()
        uv_map= geom_utils.convert_3d_to_uv_coordinates(
            uv_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # if torch.sum(uv_map != uv_map) > 0:
        #     pdb.set_trace()
        # if torch.max(uv_map) > 1.0:
        #     print('maximum index should be less that 1.0')
        #     pdb.set_trace()
        # if torch.min(uv_map) < 0.0:
        #     print('minimum value should be greater that 0.0')
        #     pdb.set_trace()

        uv_map= uv_map.permute(0, 2, 3, 1).contiguous()

        membership= self.part_membership.forward().unsqueeze(0).repeat(len(uv_map), 1, 1)
        # membership = self.part_predictor.forward(img_feat)
        predictions['cam']= camera
        predictions['cam_probs']= cam_probs
        predictions['part_transforms']= part_transforms
        predictions['delta_part_transforms']= delta_quat_transforms
        predictions['uv_map']= uv_map
        predictions['membership']= membership.to(device_id)
        predictions['iter']= real_iter

        predictions= self.post_process_predictions(predictions)
        inputs['xy_map']= torch.cat(
            [self.grid[0:1, :, :, None, 0], self.grid[0:1, :, :, None, 1]], dim=-1).unsqueeze(1)
        inputs['xy_map']= inputs['xy_map'].to(device_id)

        self.total_loss, losses= self.compute_losses(inputs, predictions)

        self.total_loss= self.total_loss.view(1,)
        return_stuff= (predictions['cam'], predictions['cam_probs'], predictions['delta_part_transforms'],
                        predictions['uv_map'], predictions['seg_mask'], predictions['project_points'],
                        predictions['project_points_cam_z'], predictions['kp_project'],
                        predictions['depth'], predictions['mask_render'], predictions['verts'],
                        inputs['img'], inputs['mask'], inputs['kp'],
                        inputs['kp_vis'], inputs['kp_valid'],
                        inputs['inds'], inputs['contour'],
                        losses['render_mask'].view(1,),
                        losses['quat_reg'].view(1,),
                        losses['rotation_reg'].view(1,),
                        losses['depth'].view(1,),
                        losses['kp'].view(1,),
                        losses['entropy'].view(1,),
                        losses['seg_mask'].view(1,),
                        losses['reproject'].view(1,),
                        losses['depth_loss_vis'],
                        losses['trans_reg'],
                        losses['coverage_mask'],
                        losses['consistency_mask'],
                        self.total_loss,)

        return return_stuff

    def post_process_predictions(self, predictions):

        device_index= self.inputs['img'].get_device()
        tensor_type= self.inputs['img'].type()
        img_size= self.inputs['img'].shape[2:]
        opts= self.opts
        b_size= len(self.inputs['img'])
        if opts.flip_train:
            true_size= b_size//2
            predictions= self.flip_predictions(
                predictions, true_size=true_size)

        real_iter= self.inputs['iter']

        verts= (self.mean_shape['verts']*1).to(device=device_index)

        if opts.warmup_deform_iter > real_iter:
            n_verts= verts.shape[0]
            predictions['verts']= verts[None, None, ...].expand(
                (b_size, opts.num_hypo_cams, n_verts, 3))
        else:
            parts_rc= self.mean_shape['parts_rc']
            parts_rc= (torch.stack(parts_rc)*1).to(device=device_index)
            parts_rc= parts_rc.unsqueeze(0).repeat(b_size, 1, 1)

            predictions['verts']= []

            for cx in range(opts.num_hypo_cams):
                part_transforms= predictions['part_transforms'][:, cx]
                verts_cx= geom_utils.apply_part_transforms(verts,
                                                            self.mean_shape['parts'],
                                                            parts_rc,
                                                            part_transforms,
                                                            predictions['membership'])
                predictions['verts'].append(verts_cx)
            predictions['verts']= torch.stack(predictions['verts'], dim=1)

        if opts.warmup_pose_iter > real_iter:
            predictions['cam_probs']= (1.0/opts.num_hypo_cams)*(torch.zeros(
                predictions['cam_probs'].shape).float() + 1).to(device_index)

        if opts.multiple_cam:
            camera= predictions['cam']
            device= camera.get_device()

            faces= (self.mean_shape['faces']*1).to(device)
            faces= faces[None, ...].repeat(b_size, 1, 1)
            verts= predictions['verts']

            mask_preds= []
            depth_preds= []
            multi_renderer_mask= self.renderers[device]['mask']
            multi_renderer_depth= self.renderers[device]['depth']
            for cx in range(opts.num_hypo_cams):
                # print(faces.get_device())
                mask_pred= multi_renderer_mask[cx].forward(
                    verts[:, cx], faces, camera[:, cx])
                mask_preds.append(mask_pred)
                depth_pred= multi_renderer_depth[cx].forward(
                    verts[:, cx], faces, camera[:, cx], depth_only=True)
                depth_preds.append(depth_pred)

            predictions['mask_render']= torch.stack(mask_preds, dim=1)
            predictions['depth']= torch.stack(depth_preds, dim=1)

            points3d= [None for _ in range(opts.num_hypo_cams)]
            predictions['project_points_cam_pred']= [
                None for _ in range(opts.num_hypo_cams)]
            predictions['project_points_cam_z']= [
                None for _ in range(opts.num_hypo_cams)]
            predictions['project_points']= [
                None for _ in range(opts.num_hypo_cams)]
            predictions['kp_project']= [
                None for _ in range(opts.num_hypo_cams)]
            predictions['verts_proj']= [
                None for _ in range(opts.num_hypo_cams)]
            kp_verts= [None for _ in range(opts.num_hypo_cams)]

            for cx in range(opts.num_hypo_cams):
                points3d[cx]= geom_utils.project_uv_to_3d(
                    self.uv2points, verts[:, cx], predictions['uv_map'])
                predictions['project_points_cam_pred'][cx]= geom_utils.project_3d_to_image(
                    points3d[cx], camera[:, cx], self.offset_z)
                predictions['project_points_cam_z'][cx]= (
                    predictions['project_points_cam_pred'][cx][..., 2] - self.cam_location[2])
                shape= (b_size, img_size[0], img_size[1])
                predictions['project_points_cam_z'][cx]= predictions['project_points_cam_z'][cx].view(
                    shape)
                shape= (b_size, img_size[0], img_size[1], 2)
                predictions['project_points'][cx]= predictions['project_points_cam_pred'][cx][..., 0:2].view(
                    shape)
                kp_verts= verts[:, cx][:, self.kp_vertex_ids, :]
                kp_project= geom_utils.project_3d_to_image(
                    kp_verts, camera[:, cx], self.offset_z)
                predictions['kp_project'][cx]= kp_project[..., 0:2].view(
                    b_size, len(self.kp_vertex_ids), -1)
                predictions['verts_proj'][cx]= geom_utils.project_3d_to_image(
                    predictions['verts'][:, cx], camera[:, cx], self.offset_z)[..., 0:2]

            predictions['verts_proj']= torch.stack(
                predictions['verts_proj'], dim=1)
            predictions['points3d']= torch.stack(points3d, dim=1)
            predictions['project_points_cam_pred']= torch.stack(
                predictions['project_points_cam_pred'], dim=1)
            predictions['project_points_cam_z']= torch.stack(
                predictions['project_points_cam_z'], dim=1)
            predictions['project_points']= torch.stack(
                predictions['project_points'], dim=1)
            predictions['kp_project']= torch.stack(
                predictions['kp_project'], dim=1)
            return predictions

    def compute_losses(self,  inputs, predictions,):
        opts= self.opts
        losses= loss_utils.code_loss(inputs, predictions, opts)
        return losses
