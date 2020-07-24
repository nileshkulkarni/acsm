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
from ...utils import cub_parse
from ...utils.visualizer import Visualizer
from ...nnutils import geom_utils
import pymesh
from ...utils import bird_vis
from ...nnutils.nmr import NeuralRenderer
from ...utils import render_utils
from ...nnutils import icn_net, geom_utils, model_utils
from ...data import pascal_imnet as pascal_imnet_data
from ...data import cub as cub_data
from ...nnutils import test_utils, mesh_geometry
from ...data import quad_imgnet as quad_imgnet_data
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
flags.DEFINE_boolean('evaluate_navie', False, 'if true visualizes things')
flags.DEFINE_boolean('evaluate_via_shape', False, 'if true evluates via pose and uv map')
flags.DEFINE_integer('seed', 0, 'seed for randomness')

flags.DEFINE_boolean('mask_dump', False, 'dump mask predictions')
flags.DEFINE_string('mask_predictions_path', None, 'Mask predictions to load')
opts = flags.FLAGS
# color_map = cm.jet(0)
kp_eval_thresholds = [0.05, 0.1, 0.2]


class KPTransferTester(test_utils.Tester):
    def define_model(self,):
        
        opts = self.opts
        self.img_size = opts.img_size
        self.offset_z = 5.0

        # membership = self.init_membership(self.mean_shape['verts'], self.mean_shape['parts'])

        init_stuff = {'alpha':self.mean_shape['alpha'],
                      'active_parts': self.part_active_state,
                      'part_axis': self.part_axis_init,
                      'kp_perm': self.kp_perm,
                      'part_perm': self.part_perm,
                      'mean_shape': self.mean_shape,
                      'cam_location': self.cam_location, 
                      'offset_z' : self.offset_z,
                      'kp_vertex_ids' : self.kp_vertex_ids,
                      'uv_sampler' : self.uv_sampler
                      }

        is_dataparallel_model = self.dataparallel_model('pred', self.opts.num_train_epoch)
        self.model = icn_net.ICPNet(opts, init_stuff )
        if is_dataparallel_model:
            self.model = torch.nn.DataParallel(self.model)
        self.load_network(self.model, 'pred', self.opts.num_train_epoch,)
        self.upsample_img_size = ((opts.img_size // 64) * (2**6),
                                  (opts.img_size // 64) * (2**6))

        self.grid = cub_parse.get_sample_grid(self.upsample_img_size).repeat(
            opts.batch_size*2, 1, 1, 1).to(self.device)

        self.offset_z = 5.0
        self.model.to(self.device)
        
        self.uv2points = cub_parse.UVTo3D(self.mean_shape)
        
        self.kp_names = self.dl.dataset.kp_names
        
        
        # self.init_render()
        self.visdom_renderer.render_mean_bird_with_uv()
        # self.triangle_loss_fn = loss_utils.LaplacianLoss(self.mean_shape['faces'].unsqueeze(0))
        self.renderer_mask = NeuralRenderer(opts.img_size)
        self.renderer_depth = NeuralRenderer(opts.img_size)

        # bird_vis.save_obj_with_texture('mean_bird', model_obj_dir,
        #                                visutil.tensor2im([self.sphere_uv_img]),
        #                                self.mean_shape_np)

        # mean_shape = {'faces': self.mean_shape_np['faces'], 
        #               'verts': self.mean_shape_np['sphere_verts'],
        #               'uv_verts' : self.mean_shape_np['uv_verts']}
        # bird_vis.save_obj_with_texture('sphere', model_obj_dir,
        #                                visutil.tensor2im([self.sphere_uv_img]),
        #                                mean_shape)
        return

    def init_render(self, ):
        opts = self.opts
        self.model_obj = pymesh.form_mesh(
            self.mean_shape['verts'].data.cpu().numpy(),
            self.mean_shape['faces'].data.cpu().numpy())

        
        sphere_obj = pymesh.form_mesh(
            self.mean_shape['sphere_verts'].data.cpu().numpy(),
            self.mean_shape['faces'].data.cpu().numpy())
        model_obj_dir = osp.join(self.save_dir, 'model')
        visutil.mkdir(model_obj_dir)
        self.model_obj_path  = osp.join(model_obj_dir, 'mean_{}.obj'.format(opts.pascal_class))
        sphere_obj_path  = osp.join(model_obj_dir, 'sphere{}.obj'.format(opts.pascal_class))
        pymesh.meshio.save_mesh(self.model_obj_path, self.model_obj)
        pymesh.meshio.save_mesh(sphere_obj_path, sphere_obj)

        nkps = len(self.kp_vertex_ids)
        self.keypoint_cmap = [cm(i * 255//nkps) for i in range(nkps)]
        
        faces_np = self.mean_shape['faces'].data.cpu().numpy()
        verts_np = self.mean_shape['sphere_verts'].data.cpu().numpy()
        uv_sampler = mesh.compute_uvsampler(
            verts_np, faces_np, tex_size=opts.tex_size)
        uv_sampler = torch.from_numpy(uv_sampler).float().cuda()
        self.uv_sampler = uv_sampler.view(-1, len(faces_np),
                                          opts.tex_size * opts.tex_size, 2)
        self.verts_uv = self.mean_shape['uv_verts']
        self.verts_obj = self.mean_shape['verts']
        
        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        self.visdom_renderer = visdom_render.VisdomRenderer(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)


        vis_rend_hr = bird_vis.VisRenderer(1024, faces_np)
        self.visdom_renderer_hr = visdom_render.VisdomRenderer(
            vis_rend_hr, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)

        # self.uv2points = cub_parse.UVTo3DHire(self.mean_shape)
        # self.uv2points = cub_parse.UVTo3D(self.mean_shape)

        vis_rend = bird_vis.VisRenderer(opts.img_size, faces_np)
        renderer_no_light = visdom_render.VisdomRenderer(
            vis_rend, self.verts_obj, self.uv_sampler, self.offset_z,
            self.mean_shape_np, self.model_obj_path, self.keypoint_cmap, self.opts)
        renderer_no_light.vis_rend.set_light_status(False)
        renderer_no_light.vis_rend.set_bgcolor((255, 255, 255))
        self.renderer_no_light = renderer_no_light

        self.sphere_uv_img = scipy.misc.imread(osp.join(opts.cachedir,'color_maps', 'sphere.png'))
        self.sphere_uv_img = torch.FloatTensor(self.sphere_uv_img)/255
        self.sphere_uv_img = self.sphere_uv_img.permute(2,0,1)
        
        return

    def init_dataset(self,):
        opts = self.opts
        if opts.dataset == 'imgnet_quad':
            self.dl = quad_imgnet_data.quad_imgnet_dataloader(opts, shuffle=True)
        else:
            if opts.pascal_class == 'bird':
                # self.dl_img1 = cub_data.cub_test_pair_dataloader(opts, 1)
                # self.dl_img2 = cub_data.cub_test_pair_dataloader(opts, 2)
                self.dl = cub_data.cub_dataloader(opts, shuffle=False)
            else:
                # self.dl_img1 = pascal_imnet_data.pascal_test_pair_dataloader(opts, 1)
                # self.dl_img2 = pascal_imnet_data.pascal_test_pair_dataloader(opts, 2)
                self.dl = pascal_imnet_data.pascal_dataloader(opts, shuffle=False)


        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.kp_perm = self.dl.dataset.kp_perm
        self.kp_perm = torch.LongTensor(self.kp_perm)
        self.preload_model_data()
        self.init_render()
        return

    def preload_model_data(self,):
        opts = self.opts
        model_dir, self.mean_shape, self.mean_shape_np = model_utils.load_template_shapes(opts)
        dpm, parts_data, self.kp_vertex_ids  = model_utils.init_dpm(self.dl.dataset.kp_names, model_dir, self.mean_shape, opts.parts_file)
        opts.nparts = self.mean_shape['alpha'].shape[1]
        self.part_active_state, self.part_axis_init, self.part_perm = model_utils.load_active_parts(model_dir, self.save_dir, dpm, parts_data, suffix='')
        return

    def set_input(self, batch):

        # batch = cub_parse.collate_pair_batch(batch)

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
        self.codes_gt['kp_valid'] = batch['kp_valid'].to(self.device)  >0.5


        self.codes_gt['contour'] = (batch['contour']).float().to(self.device)
        self.codes_gt['contour'] = (self.codes_gt['contour']/img_size - 0.5)*2


        kps_vis = self.codes_gt['kp'][..., 2] > 0
        kps_ind = (self.codes_gt['kp'] * 0.5 + 0.5) * img_size
        self.codes_gt['kps_vis'] = kps_vis
        self.codes_gt['kps_ind'] = kps_ind
        self.codes_gt['kps_vis'] = kps_vis * self.codes_gt['kp_valid']

        return
        # input_imgs = batch['img'].type(self.Tensor)
        # mask = batch['mask'].type(self.Tensor)
        # for b in range(input_imgs.size(0)):
        #     input_imgs[b] = self.resnet_transform(input_imgs[b])
        # self.inds = [k.item() for k in batch['inds']]
        # self.input_img_tensor = input_imgs.to(self.device)
        # self.mask = mask.to(self.device)
        # self.codes_gt = {}
        # self.codes_gt['kp'] = batch['kp'].type(self.Tensor).to(self.device)
        # kps_vis = self.codes_gt['kp'][..., 2] > 0
        # kps_ind = (self.codes_gt['kp'] * 0.5 + 0.5) * self.input_img_tensor.size(-1)
        # self.codes_gt['kps_vis'] = kps_vis
        # self.codes_gt['kps_ind'] = kps_ind

    def predict(self, ):
        opts = self.opts
        feed_dict = {}
        codes_gt = self.codes_gt
        feed_dict['img'] = self.input_img_tensor
        feed_dict['mask'] = self.mask.unsqueeze(1)
        feed_dict['mask_df'] = self.mask_df.unsqueeze(1)
        feed_dict['iter'] = 0
        feed_dict['kp'] = codes_gt['kp']
        feed_dict['kp_valid'] = codes_gt['kp_valid']
        feed_dict['inds'] = codes_gt['inds']
        feed_dict['contour'] = codes_gt['contour']
        # codes_pred = self.model.forward(feed_dict)
        # codes_gt = self.codes_gt
        b_size = len(feed_dict['img'])
        nverts = self.mean_shape['verts'].shape[0]
        predictions_tuple = self.model.forward(img=feed_dict['img'],
                                               mask=feed_dict['mask'],
                                               mask_df=feed_dict['mask_df'],
                                               kp=feed_dict['kp'],
                                               kp_valid=feed_dict['kp_valid'],
                                               inds=feed_dict['inds'],
                                               real_iter=0,
                                               contour = feed_dict['contour'], )
        

        codes_pred = {}
        index = 0
        codes_pred['cam'] = predictions_tuple[index]; index+=1
        codes_pred['cam_probs'] = predictions_tuple[index]; index+=1
        codes_pred['delta_part_transforms'] = predictions_tuple[index]; index+=1
        codes_pred['uv_map'] = predictions_tuple[index]; index+=1
        codes_pred['seg_mask'] = predictions_tuple[index]; index+=1
        codes_pred['project_points'] = predictions_tuple[index]; index+=1
        codes_pred['project_points_cam_z'] = predictions_tuple[index]; index+=1
        codes_pred['kp_project'] = predictions_tuple[index]; index+=1
        codes_pred['depth'] = predictions_tuple[index]; index+=1
        codes_pred['mask_render'] = predictions_tuple[index]; index+=1
        codes_pred['verts'] = predictions_tuple[index]; index+=1
        codes_pred['nmr_uv_render'] = predictions_tuple[index]; index+=1
        codes_pred['sdf_nmr_uv'] = predictions_tuple[index]; index+=1


        codes_gt['img'] = predictions_tuple[index]; index+=1
        codes_gt['mask'] = predictions_tuple[index]; index+=1
        codes_gt['kp'] = predictions_tuple[index]; index+=1
        codes_gt['kps_vis'] = predictions_tuple[index]; index+=1
        codes_gt['kp_valid'] = predictions_tuple[index]; index+=1
        codes_gt['kps_ind'] = (codes_gt['kp'] * 0.5 + 0.5) * self.img_size
        codes_gt['inds'] = predictions_tuple[index]; index+=1
        codes_gt['contour'] = predictions_tuple[index]; index+=1

        self.codes_pred = codes_pred
        self.codes_gt = codes_gt


        bsize = len(codes_gt['img'])
        camera = []
        verts = []
        kp_project_selected = []
        for b in range(bsize):
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b], dim=0).item()
            camera.append(self.codes_pred['cam'][b][max_ind])
            verts.append(self.codes_pred['verts'][b][max_ind])
            kp_project_selected.append(self.codes_pred['kp_project'][b][max_ind])
        
        camera = torch.stack(camera,)
        verts = torch.stack(verts)
        kp_project_selected = torch.stack(kp_project_selected)
        
        self.codes_pred['camera_selected'] = camera
        self.codes_pred['verts_selected'] = verts
        self.codes_pred['kp_project_selected'] = kp_project_selected
        # self.codes_pred['kps_uv_reproject'] = kps_reproject_ind_modif
        
        kps_uv = 0*self.codes_gt['kp'][:,:,0:2]
        kps_inds = codes_gt['kps_ind'].long()
        # pdb.set_trace()
        for b in range(bsize):
            kps_uv[b] = codes_pred['uv_map'][b][kps_inds[b,:, 1], kps_inds[b, :, 0], :]
        kps_3d = geom_utils.project_uv_to_3d(self.uv2points, verts, kps_uv.unsqueeze(1))
        kps_3d = kps_3d.view(kps_uv.size(0), kps_uv.size(1), 3)
        # pdb.set_trace()
        
        kps_reproject = geom_utils.orthographic_proj_withz(kps_3d, camera,  offset_z=5.0)
        kps_reproject_ind_modif = kps_reproject * 128 + 128
        kps_reproject_ind_modif = kps_reproject_ind_modif[:,:,0:2]
        
        verts_uv = self.mean_shape['uv_verts'] * 1
        verts_uv = verts_uv
        vertBack = geom_utils.project_uv_to_3d(self.uv2points, verts, verts_uv[None,...].repeat(bsize,1,1).unsqueeze(1) ) 
        vert2d = geom_utils.orthographic_proj_withz(vertBack, camera,  offset_z=5.0)

        self.codes_pred['kps_reproject_ind_modif'] = kps_reproject_ind_modif
        self.codes_pred['verts2d'] = vert2d
        
        return
    
    def find_nearest_point_on_mask(self, mask, x, y):
        img_H = mask.size(0)
        img_W = mask.size(1)
        non_zero_inds = torch.nonzero(mask)
        distances = (non_zero_inds[:, 0] - y)**2 + (non_zero_inds[:, 1] - x) ** 2
        min_dist, min_index = torch.min(distances, dim=0)
        min_index = min_index.item()
        return non_zero_inds[min_index][1].item(), non_zero_inds[min_index][0].item()


    def evaluate(self,):
        kps_pred = self.codes_pred['kp_project_selected']
        kps_gt = self.codes_gt['kp']

        kps_pred_inds  = kps_pred * 128 + 128
        kps_gt_inds = kps_gt * 128 + 128
        kps1_vis = (kps_gt_inds[:,:, 2] > 200).float()
        kps_err = torch.norm((kps_pred_inds - kps_gt_inds[...,0:2]), dim=2)

        kps_err = torch.stack([kps_err, kps1_vis], dim=2)
        kps_pred_inds = kps_pred_inds.data.cpu().numpy()
        kps_gt_inds= kps_gt_inds.data.cpu().numpy()
        kps_err = kps_err.data.cpu().numpy()
        return kps_pred_inds, kps_err, kps_gt_inds
    
    def visuals_to_save(self, total_steps):
        visdom_renderer = self.visdom_renderer
        opts = self.opts
        batch_visuals = []
        mask = self.codes_gt['mask']
        img = self.codes_gt['img']
        uv_map = self.codes_pred['uv_map']
        results_dir = osp.join(opts.result_dir, "{}".format(
            opts.split), "{}".format(total_steps))
        if not osp.exists(results_dir):
            os.makedirs(results_dir)

        camera = self.codes_pred['cam']

        for b in range(len(img)):
            visuals = {}
            visuals['ind'] = "{:04}".format(self.inds[b])

            visuals['z_img'] = visutil.tensor2im(visutil.undo_resnet_preprocess(img.data[b, None, :, :, :]))
            # pdb.set_trace()
            visuals['img_kp'] = bird_vis.draw_keypoint_on_image(visuals['z_img'], self.codes_gt['kps_ind'][
                                                                b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)

            visuals['img_kp_project'] = bird_vis.draw_keypoint_on_image(visuals['z_img'], self.codes_pred['kps_reproject_ind'][
                                                                b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            verts2d = (self.codes_pred['verts2d'].data.cpu().numpy())*128 + 128
            verts2d = verts2d[:,:,0:2].astype(int)
            visuals['img_verts_proj'] = bird_vis.draw_points_on_image(visuals['z_img'], verts2d[b])
            visuals['img_verts_proj2'] = bird_vis.draw_points_on_image(visuals['z_img']*0 + 255, verts2d[b])

            if opts.evaluate_navie:
                visuals['img_kp_modf'] = bird_vis.draw_keypoint_on_image(visuals['z_img'], self.codes_pred['kps_ind_modif'][
                                                                b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            
                
            visuals['img_rep_kp'] = bird_vis.draw_keypoint_on_image(visuals['z_img'], self.codes_pred['kps_reproject_ind_modif'][b],  self.codes_gt['kps_vis'][b], self.keypoint_cmap)
            

            visuals['z_mask'] = visutil.tensor2im(
                mask.data.repeat(1, 3, 1, 1)[b, None, :, :, :])
            visuals['uv_x'], visuals['uv_y'] = render_utils.render_uvmap(
                mask[b], uv_map[b].data.cpu())
            vis_cam_hypotheses = visdom_renderer.render_all_hypotheses(camera[b],
                                                                           probs= self.codes_pred['cam_probs'][b],
                                                                           verts = self.codes_pred['verts'][b])
            visuals.update(vis_cam_hypotheses)
            # visuals['model'] =
            # (self.render_model_using_cam(self.codes_pred['cam'][b])*255).astype(np.uint8)
            max_ind = torch.argmax(self.codes_pred['cam_probs'][b].squeeze()).item()
            
            visuals['texture_copy'] = bird_vis.copy_texture_from_img(
                mask[b], img[b], self.codes_pred['project_points'][b][max_ind])

            if not opts.multiple_cam:
                camera = camera.unsqueeze(1)
                max_ind=0

            img_ix = (torch.FloatTensor(visuals['z_img']).permute(2,0,1))/255
            visuals['tfs_a0_overlay_uvmap'] = bird_vis.sample_UV_contour(img_ix, uv_map.float()[b].cpu(), self.sphere_uv_img, mask[b].float().cpu(), real_img=True)

            visuals['tfs_a0_overlay_uvmap'] = visutil.tensor2im([visuals['tfs_a0_overlay_uvmap'].data])

            visdom_renderer.update_verts(self.codes_pred['verts'][b][max_ind])
            texture_vps = visdom_renderer.render_model_using_nmr(uv_map.data[b], img.data[b], mask.data[b],
                                                                 camera[b][max_ind], upsample_texture=True)

            visuals.update(texture_vps)

            # texture_kp = visdom_renderer.render_kps_heatmap(uv_map.data[b], self.codes_gt['kps_ind'][b], self.codes_gt[
            #     'kps_vis'][b], camera[b])
            # visuals.update(texture_kp)
            if opts.evaluate_navie:
                # pdb.set_trace()
                texture_gt_kp = visdom_renderer.render_gt_kps_heatmap( self.codes_pred['kps_uv'][b], camera[b][max_ind],
                                                                        self.codes_gt['kps_vis'][b].cpu().numpy(),)
                visuals.update(texture_gt_kp)

            uv_overlayed_model_imgs = []
            for nx in range(opts.num_hypo_cams):
                self.renderer_no_light.update_verts(self.codes_pred['verts'][b][nx])
                tex_bird_img = self.renderer_no_light.wrap_texture(self.sphere_uv_img.cuda(), camera[b][nx], True, tex_size=opts.tex_size)
                uv_overlayed_model_imgs.append(tex_bird_img)

            visuals['tfs_a0_model_uv_colored'] = visutil.image_montage(uv_overlayed_model_imgs, nrow=min(3, opts.num_hypo_cams//3 + 1))

            batch_visuals.append(visuals)
            mean_shape = {'verts' : self.codes_pred['verts'][b][max_ind].data.cpu().numpy(), 
                          'faces': self.mean_shape_np['faces'], 'uv_verts': self.mean_shape_np['uv_verts']}
                
            bird_vis.save_obj_with_texture('{:04}'.format(self.inds[b]), results_dir, visuals[
                                           'texture_img'], mean_shape)
            if opts.evaluate_navie:
                bird_vis.save_obj_with_texture('kp_gt_{:04}'.format(self.inds[b]), results_dir, texture_gt_kp[
                                               'texture_kp_img_zgt'], mean_shape)

            

        return batch_visuals

    def collect_results(self, ):
        codes_gt = self.codes_gt
        codes_pred = self.codes_pred
        kps_gt = codes_gt['kp'].data.cpu().numpy()
        kps_3d = codes_pred['kp_project_selected'].data.cpu().numpy()
        uv_map = codes_pred['uv_map'].data.cpu().numpy()
        seg_mask = codes_pred['seg_mask'].data.cpu().numpy()
        cam = codes_pred['camera_selected'].data.cpu().numpy()
        verts = codes_pred['verts_selected'].data.cpu().numpy()
        faces = self.mean_shape['faces'].data.cpu().numpy()
        img = codes_gt['img']
        img = visutil.undo_resnet_preprocess(img)
        img = img.permute(0,2,3,1).data.cpu().numpy()
        kps_ind = self.codes_gt['kps_ind'].data.cpu().numpy()
        img_size = opts.img_size

        results  = {'kps_gt': kps_gt, 
                    'uv_map': uv_map,
                    'seg_mask': seg_mask,
                    'camera': cam,
                    'verts' : verts,
                    'img' : img,
                    'inds' : self.inds,
                    'img_size' :opts.img_size,
                    'faces' : faces,
        }
        return results

    def collate_vis_data(self, batches):
        def default_collate(self, elems):
            if isinstance(elems[0], np.ndarray):
                return np.concatenate(elems)
            else:
                return np.array(elems) 

        pdb.set_trace()
        collated_data = {}
        for key in batches[0].keys():                 
            collated_data[key] = default_collate([batch[key] for batch in batches])

        return collated_data

    def test(self, ):
        opts = self.opts
        bench_stats_m1 = {'kps_gt': [], 'kps_pred': [], 'kps_err': [], 'inds': [], }


        n_iter = opts.max_eval_iter if opts.max_eval_iter > 0 else len(
            self.dl)
        result_path = osp.join(
            opts.results_dir, 'results_pck_vis_dump_{}.mat'.format(n_iter))
        print('Writing to %s' % result_path)
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        bench_stats = {}
        self.iter_index = None
        self.database = {'data':[]}
        if not osp.exists(result_path) or opts.force_run:
            from itertools import izip
            # for i, batch in enumerate(izip(self.dl_img1, self.dl_img2)):
            for i, batch in enumerate(self.dl):
                self.iter_index = i
                # if i<6:
                #     continue
                
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                self.set_input(batch)
                # pdb.set_trace()
                self.predict()
                collected_data = self.collect_results()
                self.database['data'].append(collected_data)
                # inds = self.inds.cpu().numpy()
            # self.database['data'] = self.collate_vis_data(self.database['data'])
            # pdb.set_trace()
            sio.savemat(result_path, self.database)
def main(_):
    # opts.n_data_workers = 0 opts.batch_size = 1 print = pprint.pprint
    if opts.dataset != 'imgnet_quad' or opts.quad_class != 'hippo':
        opts.batch_size = 8

    opts.results_dir = osp.join(opts.results_dir_base, opts.name,  '%s' %
                                (opts.split), 'epoch_%d' % opts.num_train_epoch)
    opts.result_dir = opts.results_dir
    if opts.pascal_class =='elephant':
        opts.dl_out_imnet = True
    else:
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
