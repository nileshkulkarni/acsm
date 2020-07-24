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
from ...utils import bird_vis
from ...nnutils.nmr import NeuralRenderer
from ...utils import render_utils
from ...nnutils import icn_net, model_utils
from ...data import objects as objects_data
from ...nnutils import test_utils
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
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_boolean('mask_dump', False, 'dump mask predictions')
flags.DEFINE_string('mask_predictions_path', None, 'Mask predictions to load')
opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class KPTransferTester(test_utils.Tester):
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
        self.upsample_img_size = (
            (opts.img_size // 64) * (2**6), (opts.img_size // 64) * (2**6)
        )

        self.grid = misc_utils.get_img_grid(
            self.upsample_img_size
        ).repeat(opts.batch_size * 2, 1, 1, 1).to(self.device)

        self.offset_z = 5.0
        self.model.to(self.device)

        self.uv2points = uv_to_3d.UVTo3D(self.mean_shape)

        self.kp_names = self.dataloader.dataset.kp_names
        return

    def init_render(self, ):
        opts = self.opts
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
        kp_project_selected = []
        for b in range(bsize):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b],
                                   dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])
            kp_project_selected.append(
                self.codes_pred['kp_project'][b][max_ind]
            )

        camera = torch.stack(camera, )
        verts = torch.stack(verts)
        kp_project_selected = torch.stack(kp_project_selected)

        self.codes_pred['camera_selected'] = camera
        self.codes_pred['verts_selected'] = verts
        self.codes_pred['kp_project_selected'] = kp_project_selected
        return

    def find_nearest_point_on_mask(self, mask, x, y):
        img_H = mask.size(0)
        img_W = mask.size(1)
        non_zero_inds = torch.nonzero(mask)
        distances = (non_zero_inds[:, 0] - y)**2 + (non_zero_inds[:, 1] - x)**2
        min_dist, min_index = torch.min(distances, dim=0)
        min_index = min_index.item()
        return non_zero_inds[min_index][1].item(
        ), non_zero_inds[min_index][0].item()

    def evaluate(self, ):
        kps_pred = self.codes_pred['kp_project_selected']
        kps_gt = self.codes_gt['kp']

        kps_pred_inds = kps_pred * 128 + 128
        kps_gt_inds = kps_gt * 128 + 128
        kps1_vis = (kps_gt_inds[:, :, 2] > 200).float()
        kps_err = torch.norm((kps_pred_inds - kps_gt_inds[..., 0:2]), dim=2)

        kps_err = torch.stack([kps_err, kps1_vis], dim=2)
        kps_pred_inds = kps_pred_inds.data.cpu().numpy()
        kps_gt_inds = kps_gt_inds.data.cpu().numpy()
        kps_err = kps_err.data.cpu().numpy()
        return kps_pred_inds, kps_err, kps_gt_inds

    def visuals_to_save(self, total_steps):
        opts = self.opts
        batch_visuals = []
        mask = self.mask
        img = self.imgs
        results_dir = osp.join(
            opts.result_dir, "{}".format(opts.split), "{}".format(total_steps)
        )
        if not osp.exists(results_dir):
            os.makedirs(results_dir)

        kps = self.kps
        kp_vis = self.kps[..., 2] > 0.5
        kps_ind = (self.kps + 1) * 0.5 * opts.img_size
        for b in range(len(img)):
            visuals = {}
            visuals['ind'] = "{:04}".format(self.inds[b])

            visuals['z_img'] = visutil.tensor2im(
                visutil.undo_resnet_preprocess(img.data[b, None, :, :, :])
            )
            # pdb.set_trace()
            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'], kps_ind[b], kp_vis[b], self.keypoint_cmap
            )

            visuals['img_kp_project'] = bird_vis.draw_keypoint_on_image(
                visuals['z_img'], self.codes_pred['kps_reproject_ind'][b],
                kp_vis[b], self.keypoint_cmap
            )
            batch_visuals.append(visuals)

        return batch_visuals

    def test(self, ):
        opts = self.opts
        bench_stats_m1 = {
            'kps_gt': [],
            'kps_pred': [],
            'kps_err': [],
            'inds': [],
        }

        n_iter = np.min(len(self.dataloader), )
        result_path = osp.join(opts.results_dir, 'results_kp_rp.mat')
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = None
        if not osp.exists(result_path) or opts.force_run:
            for i, batch in enumerate(self.dataloader):
                self.iter_index = i

                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                self.set_input(batch)

                self.predict()
                kps_reproject_ind, error_kps, kps_gt_inds = self.evaluate()
                self.codes_pred['kps_reproject_ind'] = kps_reproject_ind

                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))

                bench_stats_m1['kps_gt'].append(kps_gt_inds)
                bench_stats_m1['kps_pred'].append(kps_reproject_ind)
                bench_stats_m1['kps_err'].append(error_kps)
                bench_stats_m1['inds'].append(self.inds)

            bench_stats_m1['kps_gt'] = np.concatenate(bench_stats_m1['kps_gt'])
            bench_stats_m1['kps_pred'] = np.concatenate(
                bench_stats_m1['kps_pred']
            )
            bench_stats_m1['kps_err'] = np.concatenate(
                bench_stats_m1['kps_err']
            )
            bench_stats_m1['inds'] = np.concatenate(bench_stats_m1['inds'])
            sio.savemat(result_path, bench_stats_m1)

        else:
            bench_stats = sio.loadmat(result_path)
            bench_stats_m1 = {}
            bench_stats_m1['kps_gt'] = bench_stats['kps_gt']
            bench_stats_m1['kps_pred'] = bench_stats['kps_pred']
            bench_stats_m1['kps_err'] = bench_stats['kps_err']
            bench_stats_m1['inds'] = bench_stats['inds'][0]

        from . import kp_splits
        select_kp_ids = kp_splits.get_kp_splits(
            self.kp_names, opts.category
        )

        dist_thresholds = None
        pck_eval.run_evaluation(
            bench_stats_m1, n_iter, opts.results_dir, opts.img_size,
            self.kp_names, dist_thresholds, select_kp_ids
        )


def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    opts.batch_size = 8
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
