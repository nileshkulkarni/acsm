import numpy as np
import os
import ntpath
import time
import visdom
from . import visutil as util
from . import html
import os
from datetime import datetime
import os.path as osp
import pdb
import torch
from tf_logger import Logger

class TBVisualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.log_dir = osp.join(opt.cache_dir, 'logs')
    
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # pdb.set_trace()
        # if osp.exists(osp.join(self.log_dir, opt.name)):
        #     os.removedirs(osp.join(self.log_dir, opt.name))

        log_name = datetime.now().strftime('%H_%M_%d_%m_%Y')
        print("Logging to {}".format(log_name))
        self.display_id = opt.display_id
        self.use_html = opt.is_train and opt.use_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.viz = Logger(self.log_dir, opt.name)


    def log_grads(self, model, global_step):
        self.viz.model_param_histo_summary(model, global_step)
        return

    def log_histogram(self, logs, tag, global_step):
        self.viz.histo_summary(tag, logs[tag].data.to('cpu').numpy().reshape(-1), global_step)

    def plot_current_scalars(self, scalars, global_step):
        for key, value in scalars.items():
            self.viz.scalar_summary(key, value, global_step)
