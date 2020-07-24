from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')


import scipy.io as sio
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from absl import flags
from absl import app

from ...nnutils import test_utils
from ...data import pascal_imnet as pascal_imnet_data
from ...data import cub as cub_data
from ...nnutils import de_net, geom_utils
from ...utils import render_utils
from ...nnutils.nmr import NeuralRenderer
from ...utils import bird_vis
import pymesh
from ...nnutils import geom_utils
from ...utils.visualizer import Visualizer
from ...utils import cub_parse
from ...utils import mesh
from ...utils import visutil
from ...utils import transformations
from ...utils import visdom_render
import pdb
import json
import matplotlib.pyplot as plt
from .. import pck_eval
import pprint

"""
Script for testing on CUB.
Baselines uses our model to understand scale and translation, uses cmr to know about pose.
"""


cm = plt.get_cmap('jet')
# from matplotlib import set_cmap
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_integer('seed', 0, 'seed for randomness')
flags.DEFINE_string('mask_predictions_path', None, 'Load mask annotations')
opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class CSPTester(test_utils.Tester):

    def define_model(self,):
        opts = self.opts
        assert opts.mask_predictions_path is not None , 'need acces to predicted masks'
        self.mask_preds = sio.loadmat(opts.mask_predictions_path)

        if opts.use_unet:
            self.model = de_net.DENetComplex(opts)
        else:
            self.model = de_net.DENetSimple(opts)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)

        self.downsample_grid = cub_parse.get_sample_grid(
            (opts.output_img_size, opts.output_img_size)).repeat(1, 1, 1, 1).to(self.device)
        self.grid = cub_parse.get_sample_grid((opts.img_size, opts.img_size)).repeat(1, 1, 1, 1).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        # self.kp_names = self.dl_img1.dataset.sdset.kp_names
        self.init_render()
        return

    def init_render(self, ):
        opts = self.opts
        nkps = len(self.dl_img1.dataset.kp_names)
        self.keypoint_cmap = [cm(i * 255//nkps) for i in range(nkps)]
        return

    def init_dataset(self,):
        opts = self.opts
        if opts.pascal_class == 'bird':
            self.dl_img1 = cub_data.cub_test_pair_dataloader(opts, 1)
            self.dl_img2 = cub_data.cub_test_pair_dataloader(opts, 2)
        else:
            self.dl_img1 = pascal_imnet_data.pascal_test_pair_dataloader(opts, 1)
            self.dl_img2 = pascal_imnet_data.pascal_test_pair_dataloader(opts, 2)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if opts.pascal_class == 'horse':
            mpath = osp.join(opts.pascal_anno_path, '../models/', 'horse2', 'mean_shape.mat')
        else:
            mpath= osp.join(opts.pascal_anno_path, '../models/', opts.pascal_class, 'mean_shape.mat')
        
        self.kp_names = self.dl_img1.dataset.kp_names
        # self.mean_shape = cub_parse.load_mean_shape(mpath, self.device)
        # self.mean_shape_np = sio.loadmat(mpath)
        self.kp_perm = self.dl_img1.dataset.kp_perm
        self.kp_perm = torch.LongTensor(self.kp_perm)
        return

    def set_input(self, batch):
        opts = self.opts
        batch = cub_parse.collate_pair_batch(batch)
        input_imgs = batch['img'].type(self.Tensor)
        mask = batch['mask'].type(self.Tensor)
        for b in range(input_imgs.size(0)):
            input_imgs[b] = self.resnet_transform(input_imgs[b])
        self.inds = [k.item() for k in batch['inds']]
        self.input_img_tensor = input_imgs.to(self.device)
        self.mask = mask.to(self.device)

        self.codes_gt = {}
        self.codes_gt['mask'] = self.mask
        self.codes_gt['img'] = self.input_img_tensor
        # self.kp_uv = batch['kp_uv'].type(self.Tensor).to(self.device)
        # self.codes_gt['kp_uv'] = self.kp_uv
        self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)

        # cam_pose = batch['sfm_pose'].type(self.Tensor)
        # self.cam_pose = cam_pose.to(self.device)
        # self.codes_gt['cam_gt'] = self.cam_pose

        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (self.codes_gt['kp'] * 0.5 + 0.5) * \
            self.input_img_tensor.size(-1)
        self.codes_gt['kps_vis'] = kps_vis
        self.codes_gt['kps_ind'] = kps_ind

        return


    def predict(self, ):
        opts = self.opts
        codes_gt = self.codes_gt
        imgs = codes_gt['img']

        feed_dict = {}
        feed_dict['img'] = imgs
        img_feat = self.model.forward(feed_dict)
        codes_pred = {}
        codes_pred['mask'] = self.populate_mask_from_file()
        feat1 = img_feat[0].view(1, 3, opts.output_img_size*opts.output_img_size) ## 3 x output_img_size * output_img_size
        feat2 = img_feat[1].view(1, 3, opts.output_img_size*opts.output_img_size) ## 3 x output_img_size * output_img_size

        # mask = codes_gt['mask']
        mask = (codes_pred['mask'] > 0.5).float()[:,0,:,:]
        

        codes_pred['ds_mask'] = F.grid_sample(mask[:,None,:,:], self.downsample_grid.repeat(2,1,1,1))
        ds_mask1 = (codes_pred['ds_mask'][0] > 0.5).float()
        ds_mask2 = (codes_pred['ds_mask'][1] > 0.5).float()

        # cost_map1to2 = torch.matmul(feat1.permute(0, 2,1), feat2) ## Img1 to Img2 
        # cost_map1to2 = cost_map1to2 - 10000000*(1-ds_mask2.view(1,1,-1))
        # cost_map1to2 = F.softmax(cost_map1to2, dim=2)
        # # cost_map1to2_res = cost_map1to2.view(-1, opts.output_img_size, opts.output_img_size, opts.output_img_size, opts.output_img_size)
        # # pdb.set_trace()
        # cost_map2to1 = torch.matmul(feat2.permute(0, 2,1), feat1) ## Img2 to Img1
        # cost_map2to1 = cost_map2to1 - 10000000*(1-ds_mask1.view(1,1,-1))
        
        # cost_map2to1 = F.softmax(cost_map2to1, dim=2)
        # pdb.set_trace()
        cost_map1to2 = torch.matmul(feat1.permute(0, 2,1), feat2) ## Img1 to Img2 
        cost_map1to2 = cost_map1to2 - 10000000*(1-ds_mask2.view(1,1,-1))
        # cost_map1to2 = F.softmax(cost_map1to2, dim=2)
        # cost_map1to2_res = cost_map1to2.view(-1, opts.output_img_size, opts.output_img_size, opts.output_img_size, opts.output_img_size)
        # pdb.set_trace()
        cost_map2to1 = torch.matmul(feat2.permute(0, 2,1), feat1) ## Img2 to Img1
        cost_map2to1 = cost_map2to1 - 10000000*(1-ds_mask1.view(1,1,-1))
        
        # cost_map2to1 = F.softmax(cost_map2to1, dim=2)

        codes_pred['cm1to2'] = cost_map1to2
        codes_pred['cm2to1'] = cost_map2to1

        self.codes_pred = codes_pred
        return

    def populate_mask_from_file(self,):
        iter_index = "{:05}".format(self.iter_index)
        masks = self.mask_preds[iter_index]
        mask1 = masks['mask_1'][0,0]
        mask2 = masks['mask_2'][0,0]
        mask = np.stack([mask1, mask2])
        return torch.from_numpy(mask).float().type(self.Tensor)
        # camera1 = 

    def map_kp_img1_to_img2(self, vis_inds, kps1_ind, kps2_ind, ds_mask2, cost_map1to2):

        ## Since the cost map is on a grid of size 64*64 , 64*64. We will have to scale the kps1_ind and then upscale the transfer?

        opts = self.opts
        ds_kps1_ind = torch.clamp(kps1_ind*opts.output_img_size/opts.img_size, min=0, max=opts.output_img_size-1).long()
        # ds_kps2_ind = torch.clamp(kps2_ind*opts.output_img_size/opt.img_size, min=0, max=opts.output_img_size-1).long()
        ds_kps1_ind_flatten = ds_kps1_ind[:,0] + ds_kps1_ind[:,1]*opts.output_img_size

        
        max_val, max_inds = torch.max(cost_map1to2[0, ds_kps1_ind_flatten], dim=1)
        # min_dist = 1  - max_val
        min_dist = - max_val
        kps1_vis = kps1_ind[:,2]>200
        min_dist = min_dist + (1-kps1_vis).float()*1000
        
        ds_transfer_kps = torch.stack([max_inds - (max_inds//opts.output_img_size)*opts.output_img_size, max_inds//opts.output_img_size], dim=1)
        
        transfer_kps = ds_transfer_kps * opts.img_size/opts.output_img_size ## upsample to img size

        # pdb.set_trace()
        kp_mask = torch.zeros([len(kps2_ind)]).cuda()
        kp_mask[vis_inds] = 1
        kp_transfer_error = (transfer_kps.float() - kps2_ind[:,0:2])
        kp_transfer_error = torch.norm(kp_transfer_error, dim=1)
        return transfer_kps, torch.stack([kp_transfer_error, kp_mask, min_dist], dim=1)

    def evaluate(self, ):
        codes_pred = self.codes_pred
        codes_gt = self.codes_gt
        common_kp_indices = torch.nonzero(self.codes_gt['kp'][0, :, 2] * self.codes_gt['kp'][1, :, 2] > 0.5)
        codes_pred['common_kps'] = common_kp_indices
        mask = (codes_pred['mask'] > 0.5).float()
        kp = codes_gt['kp']
        kp_inds = codes_gt['kps_ind']
        cm1to2 = codes_pred['cm1to2']
        cm2to1 = codes_pred['cm2to1']
        transfer_kps12, error_kps12 = self.map_kp_img1_to_img2(common_kp_indices, kp_inds[0], kp_inds[1], mask[1], cm1to2)
        transfer_kps21, error_kps21 = self.map_kp_img1_to_img2(common_kp_indices, kp_inds[1], kp_inds[0],  mask[0], cm2to1)
        codes_pred['tfs_12'] = transfer_kps12
        codes_pred['tfs_21'] = transfer_kps21
        kps1 = visutil.torch2numpy(kp_inds[0])
        kps2 = visutil.torch2numpy(kp_inds[1])
        return visutil.torch2numpy(transfer_kps12), visutil.torch2numpy(error_kps12), visutil.torch2numpy(transfer_kps21), visutil.torch2numpy(error_kps21), kps1, kps2
    
    def visuals_to_save(self, total_steps):
        ## For each image, render the keypoints in 3D?
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        kps_ind = self.codes_gt['kps_ind']
        codes_pred  =self.codes_pred
        codes_gt = self.codes_gt
        ## For each image, show how keypoints transfer to location of the mask
        visuals = {}
        visuals['z_img1'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            img.data[0, None, :, :, :]))
        visuals['z_img2'] = visutil.tensor2im(visutil.undo_resnet_preprocess(
            img.data[1, None, :, :, :]))
        visuals['mask1'] = visutil.tensor2im(codes_pred['mask'][0][None,:,:,:].repeat(1,3,1,1))
        visuals['mask2'] = visutil.tensor2im(codes_pred['mask'][1][None,:,:,:].repeat(1,3,1,1))

        # visuals['gt_mask1'] = visutil.tensor2im(codes_gt['mask'][0][None,None,:,:].repeat(1,3,1,1))
        # visuals['gt_mask2'] = visutil.tensor2im(codes_gt['mask'][1][None,None,:,:].repeat(1,3,1,1))
        
        visuals_tfs = bird_vis.render_transfer_kps_imgs(self.keypoint_cmap, visuals['z_img1'], visuals['z_img2'], kps_ind[0], kps_ind[1], 
            self.codes_pred['tfs_12'], self.codes_pred['tfs_21'], self.codes_pred['common_kps'] )
        visuals.update(visuals_tfs)

        # visuals['img_kp1'] = bird_vis.draw_keypoint_on_image(visuals['z_img1'],
        #     self.codes_gt['kps_ind'][0], self.codes_gt['kps_vis'][0], self.keypoint_cmap)
        visuals['ind'] = "{:04}".format(self.inds[0])
        # visuals['img_kp2'] = bird_vis.draw_keypoint_on_image(visuals['z_img2'],
        #     self.codes_gt['kps_ind'][1], self.codes_gt['kps_vis'][1], self.keypoint_cmap)

        visuals.pop('z_img1')
        visuals.pop('z_img2')
        return [visuals]


    def test(self,):
        opts = self.opts
        bench_stats_m1 = {'kps1': [], 'kps2': [],'transfer': [], 'kps_err': [], 'pair': [], }

        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(
            self.dl_img1)
        result_path = osp.join(
            opts.results_dir, 'results_{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        if not osp.exists(result_path) or opts.force_run:
            # n_iter = len(self.dl_img1)
            from itertools import izip
            self.iter_index = None
            for i, batch in enumerate(izip(self.dl_img1, self.dl_img2)):
                self.iter_index = i
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                # batch = [batch[0], batch[0]]
                self.set_input(batch)
                self.predict()
                transfer_kps12, error_kps12, transfer_kps21, error_kps21, kps1, kps2 = self.evaluate()
                # inds = self.inds.cpu().numpy()
                if opts.visualize and (i % opts.visuals_freq == 0):
                    visualizer.save_current_results(i, self.visuals_to_save(i))
                bench_stats_m1['transfer'].append(transfer_kps12)
                bench_stats_m1['kps_err'].append(error_kps12)
                bench_stats_m1['kps1'].append(kps1)
                bench_stats_m1['kps2'].append(kps2)
                bench_stats_m1['pair'].append(
                    (self.inds[0], self.inds[1]))

                bench_stats_m1['transfer'].append(transfer_kps21)
                bench_stats_m1['kps_err'].append(error_kps21)
                bench_stats_m1['kps1'].append(kps2)
                bench_stats_m1['kps2'].append(kps1)
                bench_stats_m1['pair'].append(
                    (self.inds[1], self.inds[0]))

            bench_stats_m1['kps1'] = np.stack(bench_stats_m1['kps1'])
            bench_stats_m1['kps2'] = np.stack(bench_stats_m1['kps2'])
            bench_stats_m1['transfer'] = np.stack(bench_stats_m1['transfer'])
            bench_stats_m1['kps_err'] = np.stack(bench_stats_m1['kps_err'])
            bench_stats_m1['pair'] = np.stack(bench_stats_m1['pair'])
            bench_stats = {}
            bench_stats['m1'] = bench_stats_m1

            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)
            bench_stats_m1 = {}
            bench_stats_m1['pair'] = bench_stats['m1']['pair'][0][0]
            bench_stats_m1['kps_err'] = bench_stats['m1']['kps_err'][0][0]
            bench_stats_m1['transfer'] = bench_stats['m1']['transfer'][0][0]
            bench_stats_m1['kps1'] = bench_stats['m1']['kps1'][0][0]
            bench_stats_m1['kps2'] = bench_stats['m1']['kps2'][0][0]

        
        dist_thresholds = [1e-4, 0.5*1e-3, 1E-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
        # dist_thresholds = [1.0 , 2.0]
        from . import kp_splits
        select_kp_ids = kp_splits.get_kp_splits(self.kp_names, opts.pascal_class)
        
        pck_eval.run_evaluation(bench_stats_m1, n_iter, opts.results_dir, opts.img_size, self.kp_names, dist_thresholds, select_kp_ids)
        # json_file = osp.join(opts.results_dir, 'stats_m1_{}.json'.format(n_iter))

        # stats_m1 = pck_eval.benchmark_all_instances(kp_eval_thresholds, self.kp_names, bench_stats_m1, opts.img_size)
        # stats = stats_m1
        # print(' Method 1 | Keypoint | Median Err | Mean Err | STD Err')
        # pprint.pprint(zip(stats['kp_names'], stats['median_kp_err'], stats['mean_kp_err'], stats['std_kp_err']))
        # print('PCK Values')
        # pprint.pprint(stats['interval'])
        # pprint.pprint(stats['pck'])

        # mean_pck = {}
        # for i, thresh  in enumerate(stats['interval']):
        #     mean_pck[thresh] = []
        #     for kp_name in self.kp_names:
        #         mean_pck[thresh].append(stats['pck'][kp_name][i])

        # mean_pck = {k: np.nanmean(np.array(t)) for k,t in mean_pck.items()}
        # pprint.pprint('Mean PCK  ')
        # pprint.pprint(mean_pck)

        # with open(json_file, 'w') as f:
        #     json.dump(stats, f)


        # # dist_thresholds = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1]
        # dist_thresholds = [1E-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # stats_m1 = pck_eval.benchmark_vis_instances(kp_eval_thresholds, dist_thresholds, self.kp_names, bench_stats_m1, opts.img_size)
        # stats = stats_m1
    
        # mean_pck = {}
        # points_per_thresh = {}
        # for dx, dthresh in enumerate(dist_thresholds):
        #     mean_pck[dx] = {}
        #     for i, thresh  in enumerate(stats['interval']):
        #         mean_pck[dx][thresh] = []
        #         for kp_name in self.kp_names:
        #             mean_pck[dx][thresh].append(stats['eval_params'][dx][kp_name]['acc'][i])

        #     mean_pck[dx] = {k: np.round(np.mean(np.array(t)),4) for k,t in mean_pck[dx].items()}
        #     points_per_kp = {k:v['npoints'] for k,v in  stats['eval_params'][dx].items()}
        #     points_per_thresh[dx] = np.sum(np.array(points_per_kp.values()))

        # # pdb.set_trace()
        # print('***** Distance Thresholds ***** ')
        # pprint.pprint(dist_thresholds)
        # pprint.pprint('Mean PCK  ')
        # pprint.pprint(points_per_thresh)
        # pprint.pprint(mean_pck)

        return

    def plot_mean_var_ellipse(self, means, variances):

        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, aspect='equal')

        for ix in range(len(means)):
            ell = Ellipse(xy=(means[ix][0], means[ix][1]),
                          width=variances[ix][0], height=variances[ix][1],
                          angle=0)
            color = self.keypoint_cmap[ix] * 25
            ell.set_facecolor(color[0:3])
            ell.set_alpha(0.4)
            ax.add_artist(ell)
        ax.grid(True, which='both')
        plt.scatter(means[:, 0], means[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('on')
        for i in range(len(means)):
            ax.annotate('{}'.format(i + 1), (means[i, 0], means[i, 1]))
        plt.savefig('uv_errors.png')
        return


def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    opts.results_dir = osp.join(opts.results_dir_base, opts.name,  '%s' % (opts.split), 'epoch_%d' % opts.num_train_epoch)
    opts.result_dir = opts.results_dir
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    seed = opts.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tester = CSPTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
