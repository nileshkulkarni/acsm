# this code preprocess the input image and save to the Tensor vector.
# last modified : 2017.07.12, nashory
# requirements : tps library, MIT lecense. (https://github.com/olt/thinplatespline)

import os, sys
from os.path import join
import pdb
import numpy as np
from tps import TPS, TPSError, from_control_points
import random


def tps_warp(width, height, ctlpts, stdev):
    npts = len(ctlpts['_x'][0])
    tps_pts = []
    for i in range(npts):
        for j in range(npts):
            _varx = int((random.random()-0.5) * width * stdev)
            _vary = int((random.random()-0.5) * height * stdev)
            #tps_pts.append((ctlpts['_x'][i][j], ctlpts['_y'][i][j], _varx, _vary))
            tps_pts.append((ctlpts['_x'][i][j], ctlpts['_y'][i][j], ctlpts['_x'][i][j] + _varx, ctlpts['_y'][i][j] + _vary))

    # t = TPS(tps_pts,)
    t_back = from_control_points(tps_pts, backwards=True)
    # pdb.set_trace()
    _wfx = np.zeros((width,height), dtype='float32')
    _wfy = np.zeros((width,height), dtype='float32')
    for w in range(width):
        for h in range(height):
            _wfx[w][h] = t_back.transform(w,h)[0] - w 
            _wfy[w][h] = t_back.transform(w,h)[1] - h
    warpfield_back = {'_wfx':_wfx, '_wfy':_wfy}

    t_fwd = from_control_points(tps_pts, backwards=False)
    _wfx = np.zeros((width,height), dtype='float32')
    _wfy = np.zeros((width,height), dtype='float32')
    for w in range(width):
        for h in range(height):
            _wfx[w][h] = t_fwd.transform(w,h)[0] - w 
            _wfy[w][h] = t_fwd.transform(w,h)[1] - h
    warpfield_fwd = {'_wfx' : _wfx, '_wfy' : _wfy}

    return warpfield_back, warpfield_fwd


# create control points randomly.
def control_points(width, height, offset, npts):
    # (src_x, src_y, height_x, height_y) --> (displacement-x, displacement-y)
    if not ((width-2*offset)<npts and (height-2*offset)<npts):
        w = width - 2*offset
        h = height - 2*offset

    _x = np.zeros((npts, npts), dtype ='uint32')
    #_x = _x + np.random.randint( 0,w, _x.shape,) + offset
    _y = np.zeros((npts, npts), dtype ='uint32')
    #_y = _y + np.random.randint( 0, h, _y.shape,) + offset

    for i in range(npts):
        for j in range(npts):
            _x[i][j] = offset +  int(i*(w/float(npts-1)))
            _y[i][j] = offset +  int(j*(w/float(npts-1)))

    ctlpts = {'_x':_x, '_y':_y}
    return ctlpts


def create_random_tps(img_size=256, tpslen=1000, stdev=0.25):
    g_back_mat_x = []
    g_back_mat_y = []
    g_fwd_mat_x = []
    g_fwd_mat_y = []
    npts=4
    offset = int(img_size/16.0)
    for iter in range(tpslen):
        ctl_pts = control_points(img_size, img_size, offset, npts)
        # pdb.set_trace()
        g_back, g_fwd = tps_warp(img_size, img_size, ctl_pts, stdev)
        
        g_back_mat_x.append(g_back['_wfx'])
        g_back_mat_y.append(g_back['_wfy'])
        g_fwd_mat_x.append(g_fwd['_wfx'])
        g_fwd_mat_y.append(g_fwd['_wfy'])
        
    flow_back = {'wfx' : np.stack(g_back_mat_x).copy(),'wfy': np.stack(g_back_mat_y).copy()}
    flow_fwd = {'wfx' : np.stack(g_fwd_mat_x).copy(),'wfy': np.stack(g_fwd_mat_y).copy()}
    flow = {'backward' : flow_back, 'forward' : flow_fwd}
    return flow

if __name__=="__main__":
    create_random_tps(tpslen=1)
