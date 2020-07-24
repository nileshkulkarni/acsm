from __future__ import absolute_import, division, print_function

import scipy.io as sio
import os.path as osp
import torch
import cPickle as pkl
import pdb
import numpy as np

from ..utils import cub_parse


def load_template_shapes(opts, device_mapping=0):

    if opts.category == 'bird':
        model_dir = osp.join(opts.pascal_anno_path, '../models/', opts.category)
        mpath = osp.join(model_dir, 'mean_shape.mat')
    elif opts.category in ['horse', 'sheep', 'cow']:
        if opts.category == 'horse':
            model_dir = osp.join(opts.pascal_anno_path, '../models/', 'horse2')
            # mpath = osp.join(model_dir, 'mean_shape_hire.mat')
            mpath = osp.join(model_dir, 'mean_shape_animal2.mat')
        else:
            model_dir = osp.join(
                opts.pascal_anno_path, '../models/', opts.category
            )
            mpath = osp.join(model_dir, 'mean_shape.mat')
    else:
        model_dir = osp.join(
            opts.pascal_anno_path, '../all_models', opts.category
        )
        mpath = osp.join(model_dir, 'mean_shape.mat')

    mean_shape = cub_parse.load_mean_shape(mpath, device_mapping)
    mean_shape_np = sio.loadmat(mpath)
    return model_dir, mean_shape, mean_shape_np


def init_dpm(kp_names, model_dir, mean_shape, parts_file):
    kp_vertex_ids = []
    kp2verts_file = osp.join(model_dir, 'kp2vertex.txt')

    with open(kp2verts_file) as f:
        kp2verts = {
            l.strip().split()[0]: int(l.strip().split()[1])
            for l in f.readlines()
        }
        if kp_names is None:
            for kp_name in kp2verts.keys():
                kp_vertex_ids.append(kp2verts[kp_name])
        else:
            for kp_name in kp_names:
                kp_vertex_ids.append(kp2verts[kp_name])

    kp_vertex_ids = torch.LongTensor(kp_vertex_ids).cuda()
    # pdb.set_trace()
    partPkl = osp.join(model_dir, 'parts.pkl')
    with open(partPkl, 'rb') as f:
        dpm = pkl.load(f)

    nparts = dpm['alpha'].shape[1]
    mean_shape['alpha'] = dpm['alpha'] = torch.FloatTensor(dpm['alpha'])
    mean_shape['parts_rc'] = [
        torch.FloatTensor(dpm['nodes'][1]['rc'] * 0).cuda()
    ]  ## this is a hack. RC for the 0th part should be 0,0,0
    mean_shape['parts_rc'].extend(
        [
            torch.FloatTensor(dpm['nodes'][i]['rc']).cuda()
            for i in range(1, nparts)
        ]
    )
    mean_shape['parts'] = dpm['nodes']
    assert not (parts_file == ''), 'please specify active parts file'

    with open(parts_file, 'r') as f:
        parts_data = [l.strip().split() for l in f.readlines()]

    return dpm, parts_data, kp_vertex_ids


def load_active_parts(model_dir, save_dir, dpm, parts_data, suffix='train'):
    active_part_names = {k[0]: k[1] for k in parts_data}
    part_axis = {k[0]: np.array([int(t) for t in k[2:]]) for k in parts_data}

    assert (
        active_part_names.keys().sort() == [k['name']
                                            for k in dpm['nodes']].sort()
    ), 'part names do not match'

    with open(
        osp.join(save_dir, 'active_parts_{}.txt'.format(suffix)), 'w'
    ) as f:
        for key in active_part_names.keys():
            f.write('{} {}\n'.format(key, active_part_names[key]))

    part_active_state = []
    part_axis_init = []

    for ex, key in enumerate([k['name'] for k in dpm['nodes']]):
        part_active_state.append(active_part_names[key] == 'True')
        part_axis_init.append(part_axis[key])

    with open(osp.join(model_dir, 'mirror_transforms.txt')) as f:
        mirror_pairs = [tuple(l.strip().split()) for l in f.readlines()]
        mirror_pairs = {v1: v2 for (v1, v2) in mirror_pairs}
        part_perm = []
        name2index = {
            key: ex
            for ex, key in enumerate([k['name'] for k in dpm['nodes']])
        }

        for ex, key in enumerate([k['name'] for k in dpm['nodes']]):
            mirror_key = mirror_pairs[key]
            part_perm.append(name2index[mirror_key])
        part_perm = torch.LongTensor(part_perm)

    return part_active_state, part_axis_init, part_perm