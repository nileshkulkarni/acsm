from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pdb
from ...utils import visdom_render
from ...utils import transformations
from ...utils import visutil
from ...utils import mesh
from ...nnutils import uv_to_3d
from ...nnutils import misc as misc_utils
from ...utils.visualizer import Visualizer
from ...nnutils import geom_utils
from ...utils import bird_vis
from ...nnutils.nmr import NeuralRenderer
from ...utils import render_utils
from ...nnutils import icn_net, geom_utils, model_utils

from ...data import objects as objects_data
# from ...data import pascal_imnet as pascal_imnet_data
# from ...data import cub as cub_data
from ...nnutils import test_utils, mesh_geometry
"""
Script for testing on CUB.
Sample usage: python -m dcsm.benchmark.pascal.kp_transfer --split val --name <model_name> --num_train_epoch <model_epoch>
"""

from .. import pck_eval
from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import scipy.io as sio
import cPickle as pkl
import scipy.misc
cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_boolean('mask_dump', False, 'dump mask predictions')
flags.DEFINE_boolean('use_gt_mask', False, 'use gt mask for eval')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_string('mask_predictions_path', None, 'Mask predictions to load')
flags.DEFINE_integer(
    'num_eval_iter', 0, 'Maximum evaluation iterations. 0 => 1 epoch.'
)
# flags.DEFINE_string('kp_choose_criteria', 'head', 'seed for randomness')
opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class KPTransferTester(test_utils.Tester):
    def define_model(self, ):
        opts = self.opts
        self.img_size = opts.img_size
        self.offset_z = 5.0
        self.mask_preds = None
        if opts.mask_predictions_path is not None:
            print('populating mask for birds')
            self.mask_preds = sio.loadmat(opts.mask_predictions_path)

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
            'uv_sampler': self.uv_sampler
        }

        is_dataparallel_model = self.dataparallel_model(
            'pred', self.opts.num_train_epoch
        )
        self.model = icn_net.ICPNet(opts, init_stuff)
        if is_dataparallel_model:
            self.model = torch.nn.DataParallel(self.model)
        self.load_network(
            self.model,
            'pred',
            self.opts.num_train_epoch,
        )

        self.offset_z = 5.0
        self.model.to(self.device)
        self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)
        self.kp_names = self.dataloader.dataset.kp_names
        if opts.mask_dump:
            self.mask_preds = {}
        return

    def init_render(self, ):
        opts = self.opts
        model_obj_dir = osp.join(self.save_dir, 'model')
        visutil.mkdir(model_obj_dir)
        self.model_obj_path = osp.join(
            model_obj_dir, 'mean_{}.obj'.format(opts.pascal_class)
        )
        sphere_obj_path = osp.join(
            model_obj_dir, 'sphere{}.obj'.format(opts.pascal_class)
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
        self.verts_uv = self.mean_shape['uv_verts']
        self.verts_obj = self.mean_shape['verts']

        self.sphere_uv_img = scipy.misc.imread(
            osp.join(opts.cachedir, 'color_maps', 'sphere.png')
        )
        self.sphere_uv_img = torch.FloatTensor(self.sphere_uv_img) / 255
        self.sphere_uv_img = self.sphere_uv_img.permute(2, 0, 1)

        return

    def init_dataset(self, ):

        opts = self.opts
        if opts.category == 'bird':
            self.dataloader = objects_data.cub_data_loader(opts, )
        elif opts.category in ['horse', 'sheep', 'cow']:
            self.dataloader = objects_data.imnet_pascal_quad_data_loader(
                opts, pascal_only=True
            )
        else:
            self.dataloader = objects_data.imnet_quad_data_loader(opts, )

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.kp_perm = self.dataloader.dataset.kp_perm
        self.kp_perm = torch.LongTensor(self.kp_perm)
        self.preload_model_data()
        self.init_render()
        return

    def preload_model_data(self, ):
        opts = self.opts
        model_dir, self.mean_shape, self.mean_shape_np = model_utils.load_template_shapes(
            opts
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

    def set_input(self, batch):
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = [k.item() for k in batch['inds']]
        self.imgs = input_imgs.to(self.device)
        mask = (mask > 0.5).float()
        self.mask = mask.to(self.device)
        img_size = self.imgs.shape[-1]
        self.codes_gt = {}
        self.kps = batch['kp'].type(self.Tensor).to(self.device)
        self.codes_gt['inds'] = torch.LongTensor(self.inds).to(self.device)
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)
        return

    def predict(self, ):
        opts = self.opts
        predictions = self.model.predict(self.imgs, deform=True)
        codes_pred = predictions
        self.codes_pred = codes_pred

        bsize = len(self.imgs)
        camera = []
        verts = []
        for b in range(bsize):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b],
                                   dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])

        camera = torch.stack(camera, )
        verts = torch.stack(verts)

        self.codes_pred['camera_selected'] = camera
        self.codes_pred['verts_selected'] = verts

        if self.mask_preds is not None and not opts.mask_dump:
            self.codes_pred['seg_mask'] = self.populate_mask_from_file()
        else:
            self.dump_predictions()
        return

    def populate_mask_from_file(self, ):
        iter_index = "{:05}".format(self.iter_index)
        masks = self.mask_preds[iter_index]
        mask1 = masks['mask_1'][0, 0]
        mask2 = masks['mask_2'][0, 0]
        mask = np.stack([mask1, mask2])
        return torch.from_numpy(mask).float().type(self.Tensor)

    def dump_predictions(self, ):
        opts = self.opts
        iter_index = "{:05}".format(self.iter_index)
        if opts.mask_dump:
            mask_np = self.codes_pred['seg_mask'].data.cpu().numpy()
            mask = {}
            mask['mask_1'] = mask_np[0]
            mask['mask_2'] = mask_np[1]
            self.mask_preds[iter_index] = mask

    def find_nearest_point_on_mask(self, mask, x, y):
        img_H = mask.size(0)
        img_W = mask.size(1)
        non_zero_inds = torch.nonzero(mask)
        distances = (non_zero_inds[:, 0] - y)**2 + (non_zero_inds[:, 1] - x)**2
        min_dist, min_index = torch.min(distances, dim=0)
        min_index = min_index.item()
        return non_zero_inds[min_index][1].item(
        ), non_zero_inds[min_index][0].item()

    def map_kp_img1_to_img2(
        self,
        vis_inds,
        kps1,
        kps2,
        uv_map1,
        uv_map2,
        mask1,
        mask2,
    ):
        kp_mask = torch.zeros([len(kps1)]).cuda()
        kp_mask[vis_inds] = 1
        kps1 = kps1.long()

        kps1_vis = kps1[:, 2] > 200
        img_H = uv_map2.size(0)
        img_W = uv_map2.size(1)
        kps1_uv = uv_map1[kps1[:, 1], kps1[:, 0], :]

        kps1_3d = geom_utils.convert_uv_to_3d_coordinates(
            kps1_uv[None, None, :, :]
        )
        uv_points3d = geom_utils.convert_uv_to_3d_coordinates(
            uv_map2[None, ...]
        )
        distances3d = torch.sum(
            (kps1_3d.view(-1, 1, 3) - uv_points3d.view(1, -1, 3))**2, -1
        ).sqrt()
        distances3d = distances3d + (1 - mask2.view(1, -1)) * 1000
        distances = distances3d
        min_dist, min_indices = torch.min(distances.view(len(kps1), -1), dim=1)
        min_dist = min_dist + (1 - kps1_vis).float() * 1000
        transfer_kps = torch.stack(
            [min_indices % img_W, min_indices // img_W], dim=1
        )

        kp_transfer_error = torch.norm(
            (transfer_kps.float() - kps2[:, 0:2].float()), dim=1
        )
        return transfer_kps, torch.stack(
            [kp_transfer_error, kp_mask, min_dist], dim=1
        )

    def evaluate_m1(self, ):
        # Collect keypoints that are visible in both the images. Take keypoints
        # from one image --> Keypoints in second image.
        img_size = self.imgs.shape[-1]
        common_kp_indices = torch.nonzero(
            self.kps[0, :, 2] * self.kps[1, :, 2] > 0.5
        )

        kps_ind = (self.kps + 1) * img_size / 2
        kps_ind = kps_ind.long()

        kps = self.codes_gt['kp']  # -1 to 1
        uv_map = self.codes_pred['uv_map']
        self.codes_pred['common_kps'] = common_kp_indices
        # verts = self.codes_pred['verts']
        verts = self.mean_shape['verts'].unsqueeze(0)
        if self.opts.use_gt_mask:
            mask = (self.mask > 0.5).float()
        else:
            mask = (self.codes_pred['seg_mask'] > 0.5).float()

        transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(
            common_kp_indices,
            kps_ind[0],
            kps_ind[1],
            uv_map[0],
            uv_map[1],
            mask[0],
            mask[1],
        )
        transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(
            common_kp_indices,
            kps_ind[1],
            kps_ind[0],
            uv_map[1],
            uv_map[0],
            mask[1],
            mask[0],
        )

        kps1 = visutil.torch2numpy(kps_ind[0])
        kps2 = visutil.torch2numpy(kps_ind[1])

        self.codes_pred['tfs_12'] = transfer_kps12
        self.codes_pred['tfs_21'] = transfer_kps21
        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(
            error_kps12
        ), visutil.torch2numpy(transfer_kps21
                               ), visutil.torch2numpy(error_kps21), kps1, kps2

    def visuals_to_save(self, total_steps):
        visdom_renderer = self.visdom_renderer
        opts = self.opts
        batch_visuals = []
        uv_map = self.codes_pred['uv_map']
        results_dir = osp.join(
            opts.result_dir, "{}".format(opts.split), "{}".format(total_steps)
        )
        if not osp.exists(results_dir):
            os.makedirs(results_dir)

        camera = self.codes_pred['cam']

        for b in range(len(self.imgs)):
            visuals = {}
            visuals['ind'] = "{:04}".format(self.inds[b])

            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(
                    self.imgs.data[b, None, :, :, :]
                )
            )
            batch_visuals.append(visuals)

        mask = self.mask
        img = self.imgs
        kps_ind = (self.kps + 1) * opts.img_size / 2
        codes_pred = self.codes_pred
        codes_gt = self.codes_gt
        common_kp_indices = torch.nonzero(
            self.kps[0, :, 2] * self.kps[1, :, 2] > 0.5
        )

        visuals_tfs = bird_vis.render_transfer_kps_imgs(
            self.keypoint_cmap, batch_visuals[0]['z_img'],
            batch_visuals[1]['z_img'], kps_ind[0], kps_ind[1],
            self.codes_pred['tfs_12'], self.codes_pred['tfs_21'],
            common_kp_indices
        )
        batch_visuals[0].update(visuals_tfs)
        batch_visuals[1].update(visuals_tfs)

        return batch_visuals

    def test(self, ):
        opts = self.opts
        bench_stats_m1 = {
            'kps1': [],
            'kps2': [],
            'transfer': [],
            'kps_err': [],
            'pair': [],
        }

        result_path = osp.join(
            opts.results_dir, 'results_{}.mat'.format(opts.num_eval_iter)
        )
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = 0
        num_epochs = opts.num_eval_iter // len(self.dataloader) + 1
        if not osp.exists(result_path):
            for _ in range(num_epochs):
                for _, batch in enumerate(self.dataloader):
                    self.iter_index += 1
                    if self.iter_index % 100 == 0:
                        print(
                            '{}/{} evaluation iterations.'.format(
                                self.iter_index, opts.num_eval_iter
                            )
                        )
                    if opts.max_eval_iter > 0 and (
                        self.iter_index >= opts.max_eval_iter
                    ):
                        break
                    self.set_input(batch)
                    self.predict()
                    transfer_kps12, error_kps12, transfer_kps21, error_kps21, kps1, kps2 = self.evaluate_m1(
                    )

                    # pdb.set_trace()
                    if opts.visualize and (
                        self.iter_index % opts.visuals_freq == 0
                    ):
                        visualizer.save_current_results(
                            self.iter_index,
                            self.visuals_to_save(self.iter_index)
                        )

                    # transfer_kps12, error_kps12, transfer_kps21, error_kps21 = self.evaluate_m1_via_shape()
                    bench_stats_m1['transfer'].append(transfer_kps12)
                    bench_stats_m1['kps_err'].append(error_kps12)
                    bench_stats_m1['kps1'].append(kps1)
                    bench_stats_m1['kps2'].append(kps2)
                    bench_stats_m1['pair'].append((self.inds[0], self.inds[1]))

                    bench_stats_m1['transfer'].append(transfer_kps21)
                    bench_stats_m1['kps_err'].append(error_kps21)
                    bench_stats_m1['kps1'].append(kps2)
                    bench_stats_m1['kps2'].append(kps1)
                    bench_stats_m1['pair'].append((self.inds[1], self.inds[0]))

                    if self.iter_index > opts.num_eval_iter:
                        break

            bench_stats_m1['kps1'] = np.stack(bench_stats_m1['kps1'])
            bench_stats_m1['kps2'] = np.stack(bench_stats_m1['kps2'])
            bench_stats_m1['transfer'] = np.stack(bench_stats_m1['transfer'])
            bench_stats_m1['kps_err'] = np.stack(bench_stats_m1['kps_err'])
            bench_stats_m1['pair'] = np.stack(bench_stats_m1['pair'])
            bench_stats['m1'] = bench_stats_m1

            if opts.mask_dump:
                mask_file = osp.join(
                    opts.results_dir,
                    'mask_dump_{}.mat'.format(opts.num_eval_iter)
                )
                sio.savemat(mask_file, self.mask_preds)
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)
            bench_stats_m1 = {}
            bench_stats_m1['pair'] = bench_stats['m1']['pair'][0][0]
            bench_stats_m1['kps_err'] = bench_stats['m1']['kps_err'][0][0]
            bench_stats_m1['transfer'] = bench_stats['m1']['transfer'][0][0]
            bench_stats_m1['kps1'] = bench_stats['m1']['kps1'][0][0]
            bench_stats_m1['kps2'] = bench_stats['m1']['kps2'][0][0]

        dist_thresholds = [
            1e-4, 1e-3, 0.25 * 1e-2, 0.5 * 1e-2, 0.75 * 1e-2, 1E-2, 1E-1, 0.2,
            0.3, 0.4, 0.5, 0.6, 10
        ]
        # dist_thresholds =  [100]
        from . import kp_splits
        select_kp_ids = kp_splits.get_kp_splits(
            self.kp_names, opts.pascal_class
        )
        pck_eval.run_evaluation(
            bench_stats_m1, opts.num_eval_iter, opts.results_dir, opts.img_size,
            self.kp_names, dist_thresholds, select_kp_ids
        )


def main(_):
    opts.batch_size = 2
    opts.results_dir = osp.join(
        opts.results_dir_base, opts.name, '%s' % (opts.split),
        'epoch_%d' % opts.num_train_epoch
    )
    opts.result_dir = opts.results_dir
    opts.dl_out_imnet = False

    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tester = KPTransferTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
