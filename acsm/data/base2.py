"""
Code borrowed from https://github.com/shubhtuls/toe/blob/master/data/base.py
Base data loading class.
Should output:
    - img: B X 3 X H X W
    - mask: B X H X W
    - kp (optional): B X nKp X 2
    - sfm_pose (optional): B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import imageio
import scipy.linalg
import scipy.ndimage.interpolation
from absl import flags, app
from skimage import measure
import cv2
from scipy import ndimage
from skimage import measure
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pdb
from ..utils import image as image_utils
from ..utils import transformations

flags.DEFINE_boolean('dl_shuffle_inds', False, 'Shuffle inds')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_enum(
    'split', 'train', ['train', 'val', 'all', 'test'], 'eval split'
)
flags.DEFINE_float(
    'padding_frac', 0.05, 'bbox is increased by this fraction of max_dim'
)
flags.DEFINE_float(
    'jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim'
)

flags.DEFINE_boolean('tight_crop', False, 'Use Tight crops')
flags.DEFINE_boolean('flip_train', False, 'Mirror Images while training')
flags.DEFINE_integer('n_contour', 1000, 'N random samples from the contours')
flags.DEFINE_boolean(
    'dl_out_pascal', True, 'Use pascal (implies use keypoints)'
)
flags.DEFINE_boolean('dl_out_imnet', True, 'Use iment')
flags.DEFINE_string('pascal_class', 'horse', 'PASCAL VOC category name/ Cub')
flags.DEFINE_integer('num_kps', 12, 'Number of keypoints')


# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader
    '''
    def __init__(self, opts, filter_key=None):
        # Child class should define/load:
        # self.img_dir
        # self.anno
        # self.kp_perm (optional)
        # self.anno_sfm (optional)
        self._out_kp = False
        self._out_pose = False
        self._shuffle_inds = opts.dl_shuffle_inds
        if self._shuffle_inds:
            self._index_perm = None

        self.opts = opts
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.filter_key = filter_key
        self.n_contour = opts.n_contour

    def normalize_kp(self, kp, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack(
            [2 * (kp[:, 0] / img_w) - 1, 2 * (kp[:, 1] / img_h) - 1, kp[:, 2]]
        ).T
        new_kp = vis * new_kp

        return new_kp

    def normalize_pose(self, sfm_pose, img_h, img_w):
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        return sfm_pose

    def crop_image(self, img, mask, bbox, kp=None, sfm_pose=None):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        if (kp is not None):
            vis = kp[:, 2] > 0
            kp[vis, 0] -= bbox[0]
            kp[vis, 1] -= bbox[1]
        if sfm_pose is not None:
            sfm_pose[1][0] -= bbox[0]
            sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp=None, sfm_pose=None):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        mask_scale, _ = image_utils.resize_img(mask, scale)
        if kp is not None:
            vis = kp[:, 2] > 0
            kp[vis, :2] *= scale
        if sfm_pose is not None:
            sfm_pose[0] *= scale
            sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp=None, sfm_pose=None):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            if kp is not None:
                kp_perm = self.kp_perm
                # Flip kps.
                new_x = img.shape[1] - kp[:, 0] - 1
                kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
                if kp_perm is not None:
                    kp_flip = kp_flip[kp_perm, :]
                kp = kp_flip

            if sfm_pose is not None:
                # Flip sfm_pose Rot.
                R = transformations.quaternion_matrix(sfm_pose[2])
                flip_R = np.diag([-1, 1, 1,
                                  1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
                sfm_pose[2] = transformations.quaternion_from_matrix(
                    flip_R, isprecise=True
                )
                # Flip tx
                tx = img.shape[1] - sfm_pose[1][0] - 1
                sfm_pose[1][0] = tx

            return img_flip, mask_flip, kp, sfm_pose
        else:
            return img, mask, kp, sfm_pose

    def __len__(self):
        return self.num_imgs

    def forward_img(self, index):
        data = self.anno[index]

        img_path = osp.join(self.img_dir, str(data.rel_path))
        img = imageio.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float
        ) - 1

        if self._out_pose:
            data_sfm = self.anno_sfm[index]
            # sfm_pose = (sfm_c, sfm_t, sfm_r)
            sfm_pose = [
                np.copy(data_sfm.scale),
                np.copy(data_sfm.trans),
                np.copy(data_sfm.rot)
            ]
            sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
            sfm_rot[3, 3] = 1
            sfm_pose[2] = transformations.quaternion_from_matrix(
                sfm_rot, isprecise=True
            )
        else:
            sfm_pose = None

        if self._out_kp:
            parts = data.parts.T.astype(float)
            kp = np.copy(parts)
            vis = kp[:, 2] > 0
            # 0 indexed from 1 indexed
            kp[vis, :2] -= 1
            kp[np.logical_not(vis), :2] = 0
        else:
            kp = None

        # print(kp.shape)
        # if len(kp) == 16:
        #     pdb.set_trace()

        # Peturb bbox
        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac
            )
        else:
            bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(
            img, mask, bbox, kp=kp, sfm_pose=sfm_pose
        )

        # scale image, and mask. And scale kps.
        img, mask, kp, sfm_pose = self.scale_image(
            img, mask, kp=kp, sfm_pose=sfm_pose
        )

        # Mirror image on random.
        if self.opts.split == 'train':
            img, mask, kp, sfm_pose = self.mirror_image(
                img, mask, kp=kp, sfm_pose=sfm_pose
            )
        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        if self._out_kp:
            kp = self.normalize_kp(kp, img_h, img_w)
        if self._out_pose:
            sfm_pose = self.normalize_pose(sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return img, mask, kp, sfm_pose

    def _filter(self, elem):
        if self.filter_key is not None:
            if self.filter_key not in elem.keys():
                print('Bad filter key %s' % self.filter_key)
                import ipdb
                ipdb.set_trace()
            if self.filter_key == 'sfm_pose':
                # Return both vis and sfm_pose
                vis = elem['kp'][:, 2]
                elem = {
                    'vis': vis,
                    'sfm_pose': elem['sfm_pose'],
                }
            else:
                elem = elem[self.filter_key]
        return elem

    def _sample_contour(
        self,
        mask,
    ):
        # indices_y, indices_x = np.where(mask)
        # npoints = len(indices_y)
        contour = measure.find_contours(mask, 0)
        contour = np.concatenate(contour)
        sample_size = self.n_contour

        def offset_and_clip_contour(contour, offset, img_size):
            contour = contour + offset
            contour = np.clip(contour, a_min=0, a_max=img_size - 1)
            return contour

        offsets = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, -1],
                [0, -2],
                [1, 0],
                [2, 0],
                [-1, 0],
                [-2, 0],
                [-1, -1],
                [-2, -2],
                [1, 1],
                [2, 2],
                [-1, 1],
                [-2, 2],
                [1, -1],
                [2, -2],
            ]
        )

        new_contours = []
        for offset in offsets:
            temp_contour = offset_and_clip_contour(
                contour, offset.reshape(-1, 2), self.img_size
            )
            new_contours.append(temp_contour)

        new_contours = np.concatenate(new_contours)
        # contour_mask = mask * 0
        # new_contours = new_contours.astype(np.int)
        # contour_mask[new_contours[:,0], new_contours[:,1]] = 1
        npoints = len(new_contours)
        sample_indices = np.random.choice(
            range(npoints), size=sample_size, replace=False
        )

        # swtich x any y.

        temp = np.stack(
            [new_contours[sample_indices, 1], new_contours[sample_indices, 0]],
            axis=1
        )
        temp = temp.copy()
        return temp

    def mask_truncated_df(self, mask):
        mask_df = ndimage.distance_transform_edt(1 - mask)
        return mask_df

    # def _sample_contour(self, mask, n_samples=1000):
    #     contour = measure.find_contours(mask, 0)
    #     contour = np.concatenate(contour)
    #     sample_indices = np.random.choice(
    #         range(contour.shape[0]), size=n_samples, replace=True
    #     )
    #     # swtich x any y.
    #     samples = np.stack(
    #         [contour[sample_indices, 1], contour[sample_indices, 0]], axis=1
    #     )
    #     return 2 * (samples / mask.shape[0] - 0.5)

    def __getitem__(self, index):
        if self._shuffle_inds:
            if self._index_perm is None:
                self._index_perm = np.random.RandomState(seed=0).permutation(
                    self.num_imgs
                )
            index = self._index_perm[index]

        img, mask, kp, _ = self.forward_img(index)

        mask_df = self.mask_truncated_df(mask)
        contour = self._sample_contour(mask)
        valid = True
        if len(contour) != self.n_contour:
            valid = False

        elem = {
            'valid': valid,
            'img': img.astype(np.float32),
            'mask': mask.astype(np.float32),
            'inds': index,
            'mask_df': mask_df.astype(np.float32),
            'contour': contour.astype(np.float32),
            'kp': kp.astype(np.float32)
        }

        if self.opts.flip_train:
            flip_img = img[:, :, ::-1].copy()
            elem['flip_img'] = flip_img
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask
            elem['flip_mask_df'] = self.mask_truncated_df(flip_mask)
            elem['flip_contour'] = self._sample_contour(flip_mask)

        return self._filter(elem)


# --------- Kp + Cam Dataset --------- #
# ------------------------------------ #
class BaseKpCamDataset(BaseDataset):
    ''' 
    img, mask, kp, pose data loader
    '''
    def __init__(self, opts, filter_key=None):
        super(BaseKpCamDataset, self).__init__(opts, filter_key=filter_key)
        self._out_kp = True
        self._out_pose = True


# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(
    d_set_func,
    batch_size,
    opts,
    filter_key=None,
    shuffle=True,
    pascal_only=False
):
    dset = d_set_func(opts, filter_key=filter_key, pascal_only=pascal_only)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    '''Globe data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    # iterate over keys
    new_batch = []
    for t in batch:
        if t['valid']:
            new_batch.append(t)
        else:
            'Print, found an invalid batch'

    # batch = [t for t in batch if t is not None]
    batch = new_batch
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch