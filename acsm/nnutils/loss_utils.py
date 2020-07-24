'''
Loss building blocks.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from absl import flags
import numpy as np
import pdb
import itertools
from . import geom_utils
from ..utils import visutil
'''
depth_render : Is the depth coming from the renderer.
depth_pred : Is the depth coming due to prediction of the uv.
mask : Which pixels belong to the car?
'''


def depth_loss_fn(depth_render, depth_pred, mask):
    # return mask_loss_iou(mask_pred, mask_gt)

    # loss = torch.nn.functional.smooth_l1_loss(mask_pred, mask_gt, reduce=False)
    # loss = ((depth_render- depth_pred).pow(2) * mask)
    loss = torch.nn.functional.relu(depth_pred - depth_render).pow(2) * mask
    # loss = torch.nn.functional.relu(depth_pred-(depth_render + 1E-2)) * mask
    shape = loss.shape
    loss = loss.view(shape[0], shape[1], -1)
    loss = loss.mean(-1)
    return loss


def depth_loss_fn_vis(depth_render, depth_pred, mask):
    # return mask_loss_iou(mask_pred, mask_gt)

    # loss = torch.nn.functional.smooth_l1_loss(mask_pred, mask_gt, reduce=False)
    # loss = ((depth_render- depth_pred).pow(2) * mask)
    loss = torch.nn.functional.relu(depth_pred - depth_render) * mask
    return loss


def mask_loss_fn(mask_pred, mask_gt):
    # return mask_loss_iou(mask_pred, mask_gt)

    # loss = torch.nn.functional.smooth_l1_loss(mask_pred, mask_gt, reduce=False)
    # loss = torch.nn.functional.mse_loss(mask_pred, mask_gt, reduce=False)
    loss = ((mask_pred - mask_gt)**2)
    shape = loss.shape
    loss = loss.view(shape[0], shape[1], -1)
    loss = loss.mean(-1)
    return loss


def reproject_loss_l2(project_points, grid_points, mask, dim=1):
    non_mask_points = mask.view(mask.size(0), 1, -1).mean(2)
    # mask = mask.unsqueeze(-1).repeat(1, 1, 1, project_points.size(-1))
    mask = mask.unsqueeze(-1)
    loss = (mask * project_points - mask * grid_points)
    shape = loss.shape * 1
    loss = loss.pow(2).sum(-1).view(shape[0], shape[1], -1).mean(-1)
    loss = loss / (non_mask_points + 1E-10)
    return loss


def delta_quat_reg_loss_fn(quats):
    # angle = 2 * torch.acos(torch.clamp(quats[...,0], min=-1, max=1))
    cangle = quats[..., 0]
    # if not( torch.max(quats[...,0]) < 1 + 1E-5 and torch.min(quats[...,0]) > -1 - 1E-5):
    #     pdb.set_trace()
    return cangle.mean()


def compute_nmr_uv_loss(nmr_uvs, pred_uvs, mask):
    error = ((nmr_uvs - pred_uvs).abs()).sum(-1) * mask
    error = error.view(error.shape[0], error.shape[1], -1)
    error = error.mean(-1)
    return error


'''
features  : B x  N x D.
I would have to implement this loss by sampling, as the stuff does not fit in memory. So maybe sample 1000 features in space,
'''
'''
kp_pred : B x N x 2
kp_gt : B x N x 2
kp_vis: B x N 

'''


def reproject_kp_loss(kp_pred, kp_gt, kp_vis):
    loss = ((kp_pred - kp_gt)**2).sum(-1) * kp_vis
    loss = loss.mean(-1) / (kp_vis.mean(-1) + 1E-4)
    return loss


def expected_loss(loss, probs):
    return (loss * probs).sum(1).mean()


def code_loss(codes_gt, codes_pred, opts, laplacian_loss_fn=None):
    device = codes_gt['mask'].get_device()
    total_loss = []
    loss_factors = {}
    warmup_pose = False
    bsize = codes_gt['mask'].shape[0]
    if opts.warmup_pose_iter > codes_pred['iter']:
        # only train the pose predictor. Without training the probs.
        warmup_pose = True

    if (opts.warmup_pose_iter + 1) == codes_pred['iter']:
        print('Warmup pose phase completed. Training all losses together')

    seg_mask = codes_gt['mask'].squeeze()
    probs = codes_pred['cam_probs'].squeeze(2)

    # Reprojection Loss
    reproject_loss = reproject_loss_l2(
        codes_pred['project_points'], codes_gt['xy_map'], seg_mask.unsqueeze(1)
    )
    reproject_loss = expected_loss(reproject_loss, probs)
    if warmup_pose:
        reproject_loss = 0 * reproject_loss

    # Reproject keypoints
    kp_loss = reproject_kp_loss(
        codes_pred['kp_project'], codes_gt['kp'][:, :, 0:2].unsqueeze(1),
        codes_gt['kps_vis'].float().unsqueeze(1)
    )
    kp_loss = expected_loss(kp_loss, probs)

    if opts.pred_mask:
        seg_mask_loss = torch.nn.functional.binary_cross_entropy(
            codes_pred['seg_mask'], codes_gt['mask']
        )

    mask_loss = torch.zeros(1).mean().to(device)
    if opts.render_mask:
        render_mask_loss_fn = mask_loss_fn
        mask_loss = render_mask_loss_fn(
            codes_pred['mask_render'], codes_gt['mask']
        )
        if warmup_pose:
            mask_loss = mask_loss.sum(1).mean()
        else:
            mask_loss = expected_loss(mask_loss, probs)

    depth_loss = torch.zeros(1).mean().to(device)
    if opts.render_depth:
        renderer_depth = codes_pred['depth']
        shape = renderer_depth.shape
        renderer_depth = renderer_depth.view(
            shape[0] * shape[1],
            1,
            shape[2],
            shape[3],
        )

        project_points = codes_pred['project_points'].view(
            shape[0] * shape[1], shape[2], shape[3], 2
        )
        actual_depth_at_pixels = torch.nn.functional.grid_sample(
            renderer_depth, project_points.detach()
        )
        actual_depth_at_pixels = actual_depth_at_pixels.view(
            shape[0], shape[1], shape[2], shape[3]
        )
        # pdb.set_trace()
        depth_loss = depth_loss_fn(
            actual_depth_at_pixels, codes_pred['project_points_cam_z'],
            codes_gt['mask']
        )
        depth_loss = expected_loss(depth_loss, probs)
        if warmup_pose:
            depth_loss = 0 * depth_loss

        depth_loss_all_hypo_vis = depth_loss_fn_vis(
            actual_depth_at_pixels, codes_pred['project_points_cam_z'],
            codes_gt['mask']
        )

    if opts.multiple_cam:  # regularize rotation.
        NC2_perm = list(itertools.permutations(range(opts.num_hypo_cams), 2))
        NC2_perm = torch.LongTensor(zip(*NC2_perm)
                                    ).to(codes_pred['cam'].get_device())
        if len(NC2_perm) > 0:
            quats = codes_pred['cam'][:, :, 3:7]
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

        if warmup_pose:
            rotation_reg = 0 * rotation_reg

        dist_entropy = -1 * \
            (-torch.log(codes_pred['cam_probs'] + 1E-9)
             * codes_pred['cam_probs']).sum(1).mean()
        if warmup_pose:
            dist_entropy = 0 * dist_entropy
    mask_cov_err = torch.zeros(1).mean().to(device)
    if opts.mask_coverage:
        nverts = codes_pred['verts_proj'].shape[2]
        min_verts = []

        for bx in range(bsize):
            min_verts_cx = []
            for cx in range(opts.num_hypo_cams):
                with torch.no_grad():
                    # mask_cov_err = (codes_pred['verts_proj'][bx,cx,:,None,None,:] - codes_gt['xy_map'][0, 0,None,:,:, :])**2
                    mask_cov_err = (
                        codes_pred['verts_proj'][bx, cx, :, None, :] -
                        codes_gt['contour'][bx, None, :, :]
                    )**2
                    mask_cov_err = mask_cov_err.sum(-1)
                    _, min_indices = torch.topk(-1 * mask_cov_err, k=4, dim=0)
                    # _, min_indices = torch.min(mask_cov_err, dim=0)
                min_verts_cx.append(
                    codes_pred['verts_proj'][bx, cx][min_indices]
                )
            min_verts_cx = torch.stack(min_verts_cx, dim=0)
            min_verts.append(min_verts_cx)

        min_verts = torch.stack(min_verts)
        mask_cov_err = (min_verts - codes_gt['contour'][:, None, None, ...])**2
        mask_cov_err = mask_cov_err.sum(-1).mean(2)  # Topk dim is reduced.
        mask_cov_err = mask_cov_err
        mask_cov_err = mask_cov_err.mean()

    mask_con_err = torch.zeros(1).mean().to(device)
    if opts.mask_consistency:
        mask_con_err = codes_gt['mask_df'] * codes_pred['mask_render']
        temp = mask_con_err.view(bsize, opts.num_hypo_cams, -1)
        temp = temp.mean(2)
        mask_con_err = mask_con_err.mean()

    # delta_quat_reg_loss = 1 - \
    #     delta_quat_reg_loss_fn(codes_pred['delta_part_transforms'])

    regularize_trans = (codes_pred['part_transforms'][..., 1:4])**2
    regularize_trans = regularize_trans.sum(-1)  # 3
    regularize_trans = regularize_trans.sum(-1)  # nparts
    # regularize_trans = torch.nn.functional.max_pool1d(regularize_trans, kernel_size=opts.nparts)
    regularize_trans = regularize_trans.mean()

    quat_angle = torch.acos(
        torch.clamp(codes_pred['part_transforms'][..., 4], -1 + 1E-6, 1 - 1E-6)
    )
    regularize_quat = torch.mean(quat_angle)

    nmr_uvs_loss = torch.zeros(1).mean().to(device)
    if opts.learn_with_nmr_uvs:
        nmr_uvs_loss = compute_nmr_uv_loss(
            codes_pred['sdf_nmr_uv'], codes_pred['uv_map'].unsqueeze(1, ),
            codes_gt['mask']
        )
        nmr_uvs_loss = codes_pred['cam_probs'].sum(-1) * nmr_uvs_loss
        nmr_uvs_loss = nmr_uvs_loss.sum(-1)
        nmr_uvs_loss = nmr_uvs_loss.mean()

    loss_factors.update(
        {
            # 'temp': temp,
            'reproject': reproject_loss * opts.reproject_loss_wt,
            'kp': kp_loss * opts.kp_loss_wt,
            'render_mask': mask_loss * opts.mask_loss_wt,
            'depth': depth_loss * opts.depth_loss_wt,
            'seg_mask': opts.seg_mask_loss_wt * seg_mask_loss,
            'entropy': opts.ent_loss_wt * dist_entropy,
            'rotation_reg': opts.rot_reg_loss_wt * rotation_reg,
            'depth_loss_vis': depth_loss_all_hypo_vis,
            'trans_reg': opts.trans_reg_loss_wt * regularize_trans,
            'coverage_mask': mask_cov_err * opts.cov_mask_loss_wt,
            'consistency_mask': mask_con_err * opts.con_mask_loss_wt,
            'nmr_uvs': nmr_uvs_loss * opts.nmr_uv_loss_wt,
            'primary_quat_reg': regularize_quat * opts.quat_reg_wt,
        }
    )

    total_loss.append(loss_factors['seg_mask'])
    total_loss.append(loss_factors['reproject'])
    total_loss.append(loss_factors['kp'])
    total_loss.append(loss_factors['render_mask'])
    total_loss.append(loss_factors['depth'])
    total_loss.append(loss_factors['rotation_reg'])
    total_loss.append(loss_factors['entropy'])
    total_loss.append(loss_factors['trans_reg'])
    total_loss.append(loss_factors['coverage_mask'])
    total_loss.append(loss_factors['consistency_mask'])
    total_loss.append(loss_factors['nmr_uvs'])
    total_loss.append(loss_factors['primary_quat_reg'])

    total_loss = torch.stack(total_loss).sum()
    return total_loss, loss_factors
