from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pdb
import scipy.misc
import scipy.linalg
import scipy.ndimage.interpolation
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision
from ..utils import image as image_utils
from ..utils import transformations, visutil
from ..nnutils import geom_utils
import cv2
from scipy import ndimage
from skimage import measure
import socket
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('img_height', 320, 'image height')
flags.DEFINE_integer('img_width', 512, 'image width')
flags.DEFINE_enum(
    'split', 'train', ['train', 'val', 'all', 'test'], 'eval split'
)
flags.DEFINE_float(
    'padding_frac', 0.05, 'bbox is increased by this fraction of max_dim'
)
flags.DEFINE_float(
    'jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim'
)
flags.DEFINE_boolean('flip', True, 'Allow flip bird left right')
flags.DEFINE_boolean(
    'honest_mean_shape', False, 'Use mean shape from an honest source'
)
flags.DEFINE_boolean('tight_crop', False, 'Use Tight crops')
flags.DEFINE_boolean('flip_train', False, 'Mirror Images while training')
flags.DEFINE_boolean('cmr_mean_shape', False, 'Load cmr mean shape')
flags.DEFINE_integer(
    'number_pairs', 10000,
    'N random pairs from the test to check if the correspondence transfers.'
)
flags.DEFINE_integer('n_contour', 1000, 'N random samples from the contours')
flags.DEFINE_boolean(
    'dl_out_pascal', True, 'Use pascal (implies use keypoints)'
)
flags.DEFINE_boolean('dl_out_imnet', True, 'Use iment')
# flags.DEFINE_string('pascal_class', 'horse', 'PASCAL VOC category name/ Cub')
flags.DEFINE_integer('num_kps', 12, 'Number of keypoints')

if 'vl-fb' in socket.getfqdn():
    flags.DEFINE_string(
        'pascal_dir', '/data/home/cnris/nileshk/datasets/VOCdevkit/VOC2012/',
        'PASCAL Data Directory'
    )
    flags.DEFINE_string(
        'imnet_dir', '/data/home/cnris/nileshk/datasets/Imagenet',
        'Imnet Data Directory'
    )
elif 'umich' in socket.getfqdn():
    flags.DEFINE_string(
        'pascal_dir',
        '/Pool3/users/nileshk/data/DeformParts/VOCdevkit/VOC2012/',
        'PASCAL Data Directory'
    )
    flags.DEFINE_string(
        'imnet_dir', '/Pool3/users/nileshk/data/DeformParts/Imagenet',
        'Imnet Data Directory'
    )
    #flags.DEFINE_string(
    #    'pascal_dir', '/data/nileshk/DeformParts/VOCdevkit/VOC2012/', 'PASCAL Data Directory')
    #flags.DEFINE_string(
    #    'imnet_dir', '/data/nileshk/DeformParts/Imagenet', 'Imnet Data Directory')
else:
    pascal_dir_flags = '/scratch/nileshk/DeformParts/datasets/VOCdevkit/VOC2012/'
    imnet_dir_flags = '/scratch/nileshk/DeformParts/datasets/Imagenet'
    if osp.exists(pascal_dir_flags) and osp.exists(imnet_dir_flags):
        print('Streaming data from local scratch')
        flags.DEFINE_string(
            'pascal_dir', pascal_dir_flags, 'PASCAL Data Directory'
        )
        flags.DEFINE_string(
            'imnet_dir', imnet_dir_flags, 'Imnet Data Directory'
        )
    else:
        flags.DEFINE_string(
            'pascal_dir', '/nfs.yoda/nileshk/datasets/VOCdevkit/VOC2012/',
            'PASCAL Data Directory'
        )
        flags.DEFINE_string(
            'imnet_dir', '/nfs.yoda/nileshk/CorrespNet/datasets/Imagenet',
            'Imnet Data Directory'
        )


class BaseDataset(Dataset):
    def __init__(self, opts):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.n_contour = opts.n_contour
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.rngFlip = np.random.RandomState(0)
        self.flip_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(1),
                torchvision.transforms.ToTensor()
            ]
        )

        return

    def forward_img(self, index):
        data = self.anno[index]
        isdict = type(data) == dict
        # data_sfm = self.anno_sfm[index]
        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        # sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        # sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        # sfm_rot[3, 3] = 1
        # sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        imgnet_img_dir = self.imnet_img_dir
        pascal_img_dir = self.pascal_img_dir
        data_src = self.dataset_source[index]
        img_dir = pascal_img_dir if self.dataset_source[
            index] == 'pascal' else imgnet_img_dir

        if isdict:
            img_path = osp.join(img_dir, str(data['rel_path']))
        else:
            img_path = osp.join(img_dir, str(data.rel_path))
        img = scipy.misc.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        if isdict:
            mask = np.expand_dims(data['mask'], 2)
            bbox = np.array(
                [
                    data['bbox'][0], data['bbox'][1], data['bbox'][2],
                    data['bbox'][3]
                ], float
            ) - 1
            parts = np.array([[0.5, 0.5, -1] for _ in self.kp_names])
        else:
            mask = np.expand_dims(data.mask, 2)
            # Adjust to 0 indexing
            bbox = np.array(
                [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float
            ) - 1
            parts = data.parts.T.astype(float)

        if data_src == 'imnet':
            parts = np.concatenate([parts, parts], axis=0)[0:len(self.kp_names)]

        kp = np.copy(parts)
        kp[np.isnan(kp)] = 128
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox

        if self.opts.tight_crop:
            self.padding_frac = 0.0

        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac
            )
        else:
            bbox = image_utils.peturb_bbox(bbox, pf=self.padding_frac, jf=0)
        if self.opts.tight_crop:
            bbox = bbox
        else:
            bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, = self.crop_image(img, mask, bbox, kp, vis)

        # scale image, and mask. And scale kps.
        if self.opts.tight_crop:
            img, mask, kp, = self.scale_image_tight(img, mask, kp, vis)
        else:
            img, mask, kp, = self.scale_image(img, mask, kp, vis)

        # Mirror image on random.
        if self.opts.split == 'train':
            img, mask, kp, = self.mirror_image(img, mask, kp)

        # if self.opts.split == 'val':
        #     img, mask, kp,  = self.mirror_image(img, mask, kp)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm = self.normalize_kp(kp, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        return img, kp_norm, (mask > 0.5).astype(np.float),

    def normalize_kp(self, kp, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack(
            [2 * (kp[:, 0] / img_w) - 1, 2 * (kp[:, 1] / img_h) - 1, kp[:, 2]]
        ).T
        # sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        # sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        # sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp
        return new_kp

    def crop_image(self, img, mask, bbox, kp, vis):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]

        kp[vis, 0] = np.clip(kp[vis, 0], a_min=0, a_max=bbox[2] - bbox[0])
        kp[vis, 1] = np.clip(kp[vis, 1], a_min=0, a_max=bbox[3] - bbox[1])
        return img, mask, kp,  # sfm_pose

    def scale_image_tight(self, img, mask, kp, vis):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.img_size / bwidth
        scale_y = self.img_size / bheight

        # scale = self.img_size / float(max(bwidth, bheight))
        # pdb.set_trace()
        img_scale = cv2.resize(img, (self.img_size, self.img_size))
        # img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = image_utils.resize_img(mask, scale)

        mask_scale = cv2.resize(mask, (self.img_size, self.img_size))

        kp[vis, 0:1] *= scale_x
        kp[vis, 1:2] *= scale_y
        # sfm_pose[0] *= scale_x
        # sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp,  # sfm_pose

    def scale_image(
        self,
        img,
        mask,
        kp,
        vis,
    ):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        kp[vis, :2] *= scale
        # sfm_pose[0] *= scale
        # sfm_pose[1] *= scale

        return img_scale, mask_scale, kp,  # sfm_pose

    def mirror_image(self, img, mask, kp):
        kp_perm = self.kp_perm
        # pdb.set_trace()
        if self.rngFlip.rand(1) > 0.5 and self.flip:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # kp_uv_flip = kp_uv[kp_perm, :]
            # Flip sfm_pose Rot.
            # R = transformations.quaternion_matrix(sfm_pose[2])
            # flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            # sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            # tx = img.shape[1] - sfm_pose[1][0] - 1
            # sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip,  # kp_uv_flip, #sfm_pose
        else:
            return img, mask, kp,  # kp_uv,# sfm_pose

    def mask_truncated_df(self, mask):
        mask_df = ndimage.distance_transform_edt(1 - mask)
        return mask_df

    def sample_mask(
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

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        # index = 100
        # index = 800

        img, kp, mask = self.forward_img(index)
        mask_df = self.mask_truncated_df(mask)
        contour = self.sample_mask(mask)
        valid = True
        if len(contour) != self.n_contour:
            valid = False
        elem = {
            'valid':
            valid,
            'img':
            img,
            'kp':
            kp,
            'mask':
            mask,
            'mask_df':
            mask_df,
            'inds':
            np.array([index]),
            'kp_valid':
            np.array([self.dataset_source[index] == 'pascal']).astype(np.int),
            'contour':
            contour,
        }
        if self.opts.flip_train:
            flip_img = img[:, :, ::-1].copy()
            elem['flip_img'] = flip_img
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask
            elem['flip_mask_df'] = self.mask_truncated_df(flip_mask)
            elem['flip_contour'] = self.sample_mask(flip_mask)

        return elem


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
