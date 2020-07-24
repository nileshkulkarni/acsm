"""
Instance Correspondence net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch.nn as nn
import torch
import math
from absl import app
from absl import flags
import numpy as np
from . import net_blocks as nb
from . import geom_utils

flags.DEFINE_integer('num_hypo_cams', 8, 'number of hypo cams')
flags.DEFINE_integer('nparts', 0, 'number of parts')
flags.DEFINE_boolean('az_ele_quat', True, 'Predict camera as azi elev')
flags.DEFINE_boolean('no_trans', True, 'No trans in transforms')
flags.DEFINE_boolean(
    'single_axis_pred', True, 'predict rotation along a learnt axis'
)
flags.DEFINE_boolean(
    'delta_quat_pred', False, 'predict delta quat which can be regularized'
)
flags.DEFINE_boolean('multiple_cam', False, 'predict multiple cameras')
flags.DEFINE_float('scale_lr_decay', 0.05, 'Warmup the pose predictor')
flags.DEFINE_float('scale_bias', 1.0, 'Warmup the pose predictor')
flags.DEFINE_string('parts_file', '', 'active parts file name')

flags.DEFINE_float('az_euler_range', 30, 'Warmup the part transform')
flags.DEFINE_float('el_euler_range', 60, 'Warmup the part transform')
flags.DEFINE_float('cyc_euler_range', 60, 'Warmup the pose predictor')


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)

        return quat

    def initialize_to_zero_rotation(self, ):
        nb.net_init(self.pred_layer)
        self.pred_layer.bias = nn.Parameter(
            torch.FloatTensor([1, 0, 0,
                               0]).type(self.pred_layer.bias.data.type())
        )
        return


class QuatPredictorSingleAxis(nn.Module):
    def __init__(
        self,
        nz_feat,
        nz_rot=2,
    ):
        super(QuatPredictorSingleAxis, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.axis = nn.Parameter(torch.FloatTensor([1, 0, 0]))
        self.counter = 0

    def forward(self, feat):
        vec = self.pred_layer.forward(feat)
        vec = torch.nn.functional.normalize(vec)
        angle = torch.atan2(vec[:, 1], vec[:, 0]).unsqueeze(-1)
        self.axis.data = torch.nn.functional.normalize(self.axis.unsqueeze(0)
                                                       ).squeeze(0).data
        axis = self.axis.unsqueeze(0).repeat(len(angle), 1)
        quat = geom_utils.axang2quat(angle, axis)

        # self.counter += 1
        # if self.counter % 20==0:
        #     print("{}, {}".format(self.axis.data, angle))
        return quat

    def initialize_to_zero_rotation(self, ):
        nb.net_init(self.pred_layer)
        self.pred_layer.bias = nn.Parameter(
            torch.FloatTensor([1, 0]).type(self.pred_layer.bias.data.type())
        )
        return


class QuatPredictorAzEle(nn.Module):
    def __init__(self, nz_feat, nz_rot=3, euler_range=[30, 20, 20]):
        super(QuatPredictorAzEle, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 3)
        self.axis = torch.eye(3).float()
        self.euler_range = [np.pi / 180 * k for k in euler_range]
        return

    def forward(self, feat):
        axis = self.axis.to(feat.get_device())
        angles = 0.1 * self.pred_layer.forward(feat)
        angles = torch.tanh(feat)
        az_range = self.euler_range[0]
        el_range = self.euler_range[1]
        cyc_range = self.euler_range[2]
        azimuth = az_range * angles[..., 0]
        elev = el_range * (angles[..., 1])
        cyc_rot = cyc_range * (angles[..., 2])
        q_az = self.convert_ax_angle_to_quat(axis[1], azimuth)
        q_el = self.convert_ax_angle_to_quat(axis[0], elev)
        q_cr = self.convert_ax_angle_to_quat(axis[2], cyc_rot)
        quat = geom_utils.hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
        quat = geom_utils.hamilton_product(q_cr.unsqueeze(1), quat)
        return quat.squeeze(1)

    def convert_ax_angle_to_quat(self, ax, ang):
        qw = torch.cos(ang / 2)
        qx = ax[0] * torch.sin(ang / 2)
        qy = ax[1] * torch.sin(ang / 2)
        qz = ax[2] * torch.sin(ang / 2)
        quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    def initialize_to_zero_rotation(self, ):
        nb.net_init(self.pred_layer)
        return


class ScalePredictor(nn.Module):
    def __init__(
        self,
        nz,
        scale_lr_decay=0.2,
        scale_bias=1.0,
    ):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.scale_bias = scale_bias
        self.scale_lr_decay = scale_lr_decay

    def forward(self, feat):
        scale = self.scale_lr_decay * self.pred_layer.forward(
            feat
        ) + self.scale_bias
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """
    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        return trans


class Transform(nn.Module):
    def __init__(self, nz_input, no_trans, rotation_axis):
        super(Transform, self).__init__()
        self.fc_layer = nb.fc_stack(nz_input, nz_input, 2)
        self.no_trans = no_trans
        self.quat_predictor = QuatPredictorSingleAxis(nz_input)

        self.prob_predictor = nn.Linear(nz_input, 1)
        self.scale_predictor = ScalePredictor(nz_input)
        self.trans_predictor = TransPredictor(
            nz_input, orth=False
        )  ## makes it 3D dimensional.

    def forward(self, feat):
        feat = self.fc_layer(feat)
        quat_pred = self.quat_predictor.forward(feat)
        prob = self.prob_predictor(feat)
        scale = self.scale_predictor.forward(feat)
        trans = 0.1 * self.trans_predictor.forward(feat)
        if self.no_trans:
            trans = trans * 0

        return torch.cat([quat_pred, prob, scale, trans], dim=1)

    def init_quat_module(self, ):
        self.quat_predictor.initialize_to_zero_rotation()


class Camera(nn.Module):
    def __init__(
        self,
        nz_input,
        az_ele_quat,
        scale_lr_decay,
        scale_bias,
        euler_range=None,
    ):
        super(Camera, self).__init__()
        self.fc_layer = nb.fc_stack(nz_input, nz_input, 2)

        if az_ele_quat:
            assert euler_range is not None, 'requires euler range'
            self.quat_predictor = QuatPredictorAzEle(
                nz_input, euler_range=euler_range
            )
        else:
            self.quat_predictor = QuatPredictor(nz_input)

        self.prob_predictor = nn.Linear(nz_input, 1)
        self.scale_predictor = ScalePredictor(
            nz_input, scale_lr_decay=scale_lr_decay, scale_bias=scale_bias
        )
        self.trans_predictor = TransPredictor(nz_input)

    def forward(self, feat):
        feat = self.fc_layer(feat)
        quat_pred = self.quat_predictor.forward(feat)
        prob = self.prob_predictor(feat)
        scale = self.scale_predictor.forward(feat)
        trans = self.trans_predictor.forward(feat)
        return torch.cat([quat_pred, prob, scale, trans], dim=1)

    def init_quat_module(self, ):
        self.quat_predictor.initialize_to_zero_rotation()


'''
Predict camera hypothesis global transform, along with part cameras.
'''


class MultiCamPredictor(nn.Module):
    def __init__(
        self, nz_feat, num_cams, scale_lr_decay, scale_bias, no_trans,
        part_init, euler_range
    ):
        '''
            nc_input: 
            ns_input:
            nz_channels:
            nz_feat:
            num_cams:
            aze_ele_quat: Use euler angle predictions for camera
            scale_lr_decay: scale learning rate decay
            scale_bias: Scale bias
            nparts: number of parts
            no_trans: use translation in parts
            single_axis_pred: use single axis predictions for quat in transforms
            delta_quat_pred: predict delta quat for the quat in  transforms
            active_parts: number of active parts
        '''
        super(MultiCamPredictor, self).__init__()
        self.num_cams = num_cams
        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.prob_predictor = nn.Linear(nz_feat, num_cams)
        self.camera_predictor = nn.ModuleList(
            [
                Camera(
                    nz_input=nz_feat,
                    az_ele_quat=True,
                    scale_lr_decay=scale_lr_decay,
                    scale_bias=scale_bias,
                    euler_range=euler_range,
                ) for i in range(num_cams)
            ]
        )

        nb.net_init(self)

        for cx in range(num_cams):
            self.camera_predictor[cx].init_quat_module()

        ## Multiple transform predictors.
        self.transform_predictors = nn.ModuleList(
            [
                TransformPredictor(
                    nz_feat=nz_feat, no_trans=no_trans, part_init=part_init
                ) for i in range(num_cams)
            ]
        )

        base_rotation = torch.FloatTensor([0.9239, 0, 0.3827, 0]
                                          ).unsqueeze(0).unsqueeze(0)  ##pi/4
        base_bias = torch.FloatTensor([0.7071, 0.7071, 0,
                                       0]).unsqueeze(0).unsqueeze(0)
        self.cam_biases = [base_bias]
        for i in range(1, self.num_cams):
            self.cam_biases.append(
                geom_utils.hamilton_product(
                    base_rotation, self.cam_biases[i - 1]
                )
            )

        self.cam_biases = torch.stack(self.cam_biases).squeeze()
        return

    def forward(self, feat):
        feat = self.fc(feat)

        cameras = []
        for cx in range(self.num_cams):
            cameras.append(self.camera_predictor[cx].forward(feat))
        cameras = torch.stack(cameras, dim=1)

        quats = cameras[:, :, 0:4]
        scale = cameras[:, :, 5:6]
        trans = cameras[:, :, 6:8]

        prob_logits = cameras[:, :, 4]
        camera_probs = nn.functional.softmax(prob_logits, dim=1)
        cam_biases = self.cam_biases.to(feat.get_device())
        bias_quats = cam_biases.unsqueeze(0).repeat(len(quats), 1, 1)
        new_quats = geom_utils.hamilton_product(quats, bias_quats)
        cam = torch.cat(
            [scale, trans, new_quats,
             camera_probs.unsqueeze(-1)], dim=2
        )

        part_transforms = []

        for cx in range(self.num_cams):
            part_transform = self.transform_predictors[cx].forward(feat)
            part_transforms.append(part_transform)

        part_transforms = torch.stack(part_transforms, dim=1)
        return cam, part_transforms


class SingleCamPredictor(nn.Module):
    def __init__(
        self,
        nz_feat,
        scale_lr_decay,
        scale_bias,
        no_trans,
        part_init,
    ):
        super(SingleCamPredictor, self).__init__()

        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.camera_predictor = Camera(
            nz_input=nz_feat,
            az_ele_quat=False,
            scale_lr_decay=scale_lr_decay,
            scale_bias=scale_bias,
        )
        nb.net_init(self)
        self.transform_predictor = TransformPredictor(
            nz_feat=nz_feat, no_trans=no_trans, part_init=part_init
        )
        return

    def forward(self, feat):
        feat = self.fc(feat)
        cameras = self.camera_predictor.forward(feat)
        quats = cameras[..., 0:4]
        scale = cameras[..., 5:6]
        trans = cameras[..., 6:8]

        cam = torch.cat([scale, trans, quats], dim=-1)

        part_transforms = self.transform_predictor.forward(feat)
        return cam, part_transforms


class TransformPredictor(nn.Module):
    def __init__(
        self,
        part_init,
        nz_feat,
        no_trans,
    ):
        super(TransformPredictor, self).__init__()

        self.nparts = nparts = len(part_init['part_axis'])
        self.part_init = part_init
        self.fc = nb.fc_stack(nz_feat, nz_feat, 2, use_bn=False)
        self.part_transforms_pred = nn.ModuleList(
            [
                Transform(
                    nz_feat,
                    no_trans,
                    rotation_axis=part_init['part_axis'][px],
                ) for px in range(nparts)
            ]
        )
        nb.net_init(self)
        for px in range(nparts):
            self.part_transforms_pred[px].init_quat_module()

        return

    def forward(self, feats):
        feat = self.fc(feats)
        all_part_transforms = [[] for _ in range(self.nparts)]

        for px in range(self.nparts):
            part_transform_pred = self.part_transforms_pred[px]
            all_part_transforms[px] = part_transform_pred.forward(feat)

        all_part_transforms = torch.stack(all_part_transforms, dim=1)
        ## remove the prob term from here.
        all_part_transforms = torch.cat(
            [
                all_part_transforms[..., 5:6],  ## scale
                all_part_transforms[..., 6:9],  ## trans
                all_part_transforms[..., 0:4],
            ],  ## quat
            dim=-1
        )

        active_parts = self.part_init['active_parts']

        for px in range(len(active_parts)):
            if active_parts[px] == False:
                all_part_transforms[:, px, :] = 0  ## making it zero trans
                all_part_transforms[:, px,
                                    4] = 1  ## making it zero quat (no rotation)

        return all_part_transforms


def reflect_cam_pose(cam_pose):
    new_cam_pose = cam_pose * torch.FloatTensor(
        [1, -1, 1, 1, 1, -1, -1]
    ).view(1, 1, -1).to(device=cam_pose.device.index)
    return new_cam_pose