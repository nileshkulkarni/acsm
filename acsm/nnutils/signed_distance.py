from __future__ import print_function, division, absolute_import
from threading import Thread
import scipy.ndimage
import numpy as np
from torch.multiprocessing import Pool, Process
import pdb
import os
import torch


def worker_distance_transform(args):
    bid = args[0]
    image = args[1]
    return_indices = args[2]
    # print('{} , {} '.format(bid, os.getpid()))
    image = image.numpy()
    dist, dist_indices = scipy.ndimage.distance_transform_edt(
        image, return_indices=return_indices)
    dist_indices = torch.LongTensor(dist_indices)
    return dist_indices


class BatchedSignedDistance:
    '''
        A batched version of computing signed distance in parallel.
    '''

    def __init__(self, num_workers=4, return_indices=True):
        self.num_workers = num_workers
        self.return_indices = return_indices
        return

    def forward(self, images, ):
        parameters = []
        num_workers = self.num_workers
        pool = Pool(num_workers)
        for bx in range(len(images)):
            bx_params = [bx, images[bx], True]
            parameters.append(bx_params, )

        predictions = pool.map(
            worker_distance_transform, parameters)
        predictions = torch.stack(predictions)
        pool.close()
        pool.join()
        return predictions


if __name__ == "__main__":
    batchedSignedDist = BatchedSignedDistance(num_workers=1)

    a = np.array(([0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 0, 0]))
    data = torch.FloatTensor(np.stack([1-a, 1-a, 1-a, 1-a], axis=0))

    outputs = batchedSignedDist.forward(data)
    outputs = batchedSignedDist.forward(data)
    pdb.set_trace()
    print ('Done')
