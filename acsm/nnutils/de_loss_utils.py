from __future__ import absolute_import, division, print_function
import pdb
import torch
import torch.nn.functional as F
from absl import flags

flags.DEFINE_float('nll_loss_wt', 1, 'Negative Loss Weight')
flags.DEFINE_float('exp_loss_wt', 10, 'Expected Distance Loss Weight')

def code_loss(codes_pred, codes_gt, opts):

    loss_factors = {}
    total_loss = []
    bsize, feat_dim, H, W = codes_pred['feat_u'].shape
    phiu =  codes_pred['feat_u']
    phiv = codes_pred['feat_v']
    phiu = phiu.view(bsize, feat_dim, H*W)
    phiv = phiv.view(bsize, feat_dim, H*W)
    
    cost_map = torch.matmul(phiu.permute(0,2,1), phiv )  ## B x H*W x H*W
    cost_map = F.softmax(cost_map, dim=2) ## cost map is along dimension 2. Where things sum to 2
    
    u2v_mapping = codes_gt['guv_adj']  ## B x H x W x 2
    u2v_mapping_flatten = u2v_mapping.view(bsize*H*W, 1,1, 2)

    # pdb.set_trace()
    max_cost, max_ind = torch.max(cost_map, dim=2)
    max_ind_x = max_ind - (max_ind//opts.output_img_size)*opts.output_img_size
    max_ind_y = max_ind//opts.output_img_size
  
    loss_factors['pred_u2v_map'] = torch.stack([max_ind_x, max_ind_y ],dim=2)
    loss_factors['pred_u2v_map'] = loss_factors['pred_u2v_map'].view(bsize, H, W, 2)

    probabilities = F.grid_sample(cost_map.view(bsize*H*W, 1, H, W), u2v_mapping_flatten)
    probabilities = probabilities.view(bsize,-1) ## B x H*W

    mask_u = (codes_gt['ds_mask_u']>0.5).float()
    mask_v = (codes_gt['ds_mask_v']>0.5).float()
    
    log_probabilities = (1E-12 + probabilities).log()
    log_probabilities = (log_probabilities* mask_u.view(bsize, -1))
    log_probabilities = log_probabilities.sum(1)/(H*W)
    loss_nll = -1*log_probabilities

    loss_factors['nll'] = opts.nll_loss_wt*loss_nll.mean()
    total_loss.append(loss_factors['nll'])

    v = codes_gt['downsample_grid'].view(bsize,1, H*W, 2)
    gu = u2v_mapping.view(bsize, H*W, 1 , 2)
    v = v * mask_v.view(bsize, 1, H*W, 1)
    gu = gu * mask_u.view(bsize, H*W, 1, 1)
    # pdb.set_trace()
    expected_distances = torch.norm(gu - v, dim=3).pow(0.5)*cost_map
    expected_distances = expected_distances.sum(2).sum(1)/(H*W)
    
    loss_factors['exp_dist']  = opts.exp_loss_wt * expected_distances.mean()
    total_loss.append(loss_factors['exp_dist'])

    total_loss = torch.stack(total_loss).sum()
    return total_loss, loss_factors