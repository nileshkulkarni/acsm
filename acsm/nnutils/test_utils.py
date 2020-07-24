"""
Generic Testing Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags

import scipy.misc

from ..utils.visualizer import Visualizer

#-------------- flags -------------#
#----------------------------------#
# Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cachedir', cache_path, 'Cachedir')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
# Set it as split in dataloader
# flags.DEFINE_string('eval_set', 'val', 'which set to evaluate on')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')

flags.DEFINE_integer('batch_size', 2, 'Size of minibatches')
flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')

# Flags for logging and snapshotting
flags.DEFINE_string(
    'checkpoint_dir', osp.join(cache_path, 'snapshots'),
    'Directory where networks are saved'
)

flags.DEFINE_boolean('use_gt_cam', False, 'uses sfm camera')
flags.DEFINE_boolean(
    'use_gt_quat', False,
    'Use only the GT orientation. Only valid if used with pred_cam'
)

flags.DEFINE_string(
    'results_dir_base', osp.join(cache_path, 'evaluation'),
    'Directory where evaluation results will be saved'
)
flags.DEFINE_string('results_dir', '', 'This gets set automatically now')
flags.DEFINE_string('result_dir', '', 'This gets set automatically now')

flags.DEFINE_integer(
    'max_eval_iter', 0, 'Maximum evaluation iterations. 0 => 1 epoch.'
)
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')

flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')
flags.DEFINE_boolean('texture', True, 'if true uses texture!')

# Unused
flags.DEFINE_boolean(
    'save_visuals', False, 'Whether to save intermittent visuals'
)
flags.DEFINE_boolean('force_run', False, 'Run even if the results file exists')
flags.DEFINE_integer(
    'visuals_freq', 50, 'Save visuals every few forward passes'
)

flags.DEFINE_string(
    'results_vis_dir', osp.join(cache_path, 'results_vis'),
    'Directory where intermittent results will be saved'
)
flags.DEFINE_string(
    'results_eval_dir', osp.join(cache_path, 'evaluation'),
    'Directory where evaluation results will be saved'
)

flags.DEFINE_string('dataset', 'globe', 'dataset source')


#-------- tranining class ---------#
#----------------------------------#
class Tester():
    def __init__(self, opts):
        self.opts = opts
        if opts.split == 'train':
            print('*** Caution : You are testing on the train set *** ')

        self.vis_iter = 0
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (
            self.gpu_id is not None
        ) else torch.Tensor
        # the trainer can optionally reset this every iteration during set_input call
        self.invalid_batch = False
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.cam_location = [0, 0, -2.732]
        log_file = os.path.join(self.save_dir, 'opts_testing.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))
        self.device = 'cuda:{}'.format(opts.gpu_id)
        self.offset_z = 5.0

    def dataparallel_model(self, network_label, epoch_label, network_dir=None):

        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        state_dict = torch.load(save_path)
        return state_dict.keys()[0].startswith('module.')

    # helper loading function that can be used by subclasses

    def load_network(
        self, network, network_label, epoch_label, network_dir=None
    ):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        print('Loading weight from ' + save_path)
        network.load_state_dict(torch.load(save_path))
        return

    def save_current_visuals(self):
        visuals = self.get_current_visuals()
        imgs_dir = osp.join(
            self.opts.results_vis_dir, 'vis_iter_{}'.format(self.vis_iter)
        )
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        for k in visuals:
            img_path = osp.join(imgs_dir, k + '.png')
            scipy.misc.imsave(img_path, visuals[k])
        self.vis_iter += 1

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_testing(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()

    def test(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError
