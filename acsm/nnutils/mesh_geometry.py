"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""

import torch
import pdb
from torch import nn
import numpy as np
from . import geom_utils
from ..utils import cub_parse


def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0
    d2 = np.copy(tri[2, :]) - p0
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
        inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return False, 0

    # inv_mat = np.linalg.inv(mat)

    a_b_mg = -1 * np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and (
        (a_b_mg[0] + a_b_mg[1]) <= 1
    ) and (a_b_mg[2] < 0)
    if is_valid:
        # pdb.set_trace()
        return True, -a_b_mg[2] * d, a_b_mg
    else:
        return False, 0, 0


def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    for nv in range(verts.shape[0]):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out


def compute_barycentric_coordinates(uv_verts, uv_points):

    verts = geom_utils.convert_uv_to_3d_coordinates(uv_verts)
    points = geom_utils.convert_uv_to_3d_coordinates(uv_points)

    bsize = len(uv_verts)
    new_points = [[] for _ in range(bsize)]
    for bx in range(bsize):
        for px, point in enumerate(points[bx]):
            valid, intersect, _ = triangle_direction_intersection(
                verts[bx, px].cpu().numpy(),
                point.cpu().numpy()
            )
            new_points[bx].append(torch.FloatTensor(intersect).cuda())
        new_points[bx] = torch.stack(new_points[bx])

    # pdb.set_trace()
    points = torch.stack(new_points)

    vertA = verts[..., 0, :]
    vertB = verts[..., 1, :]
    vertC = verts[..., 2, :]

    AB = vertB - vertA
    AC = vertC - vertA
    BC = vertC - vertB

    AP = points - vertA
    BP = points - vertB
    CP = points - vertC

    areaBAC = torch.norm(torch.cross(AB, AC, dim=-1), dim=-1)
    areaBAP = torch.norm(torch.cross(AB, AP, dim=-1), dim=-1)
    areaCAP = torch.norm(torch.cross(AC, AP, dim=-1), dim=-1)
    areaCBP = torch.norm(torch.cross(BC, BP, dim=-1), dim=-1)

    w = areaBAP / areaBAC
    v = areaCAP / areaBAC
    u = areaCBP / areaBAC
    barycentric_coordinates = torch.stack([u, v, w], dim=-1)

    barycentric_coordinates = torch.nn.functional.normalize(
        barycentric_coordinates, p=1, dim=-1
    )
    return barycentric_coordinates


'''
This method does a brute force search over all the faces to determine if a face is within
    template_shape : preloaded from mat file
    verts:  B x N_{v} x 3
    uv: B x N x 2
'''


def project_uv_to_3d_brute(template_shape, verts, uv_crds, get_bary=False):
    B, _, _ = verts.shape
    verts_sphere = template_shape['sphere_verts']
    uv_verts = template_shape['uv_verts']
    faces = template_shape['faces']
    faces_np = template_shape['faces'].data.cpu().numpy()
    uv_verts3d = geom_utils.convert_uv_to_3d_coordinates(uv_verts)
    uv_verts3d_np = uv_verts3d.data.cpu().numpy()
    uv_map3d = geom_utils.convert_uv_to_3d_coordinates(uv_crds)
    uv_map3d_np = uv_map3d.data.cpu().numpy()

    uv_verts = uv_verts.unsqueeze(0).repeat(B, 1, 1)
    uv_faceids = [[] for _ in range(len(uv_map3d.view(-1, 3)))]
    for ix, uv3d in enumerate(uv_map3d_np.reshape(-1, 3)):
        for fx, face in enumerate(faces_np):
            tri = uv_verts3d_np[face]
            valid, intersect, wi = triangle_direction_intersection(tri, uv3d)

            if valid:
                uv_faceids[ix].append(fx)
                break
            # else:
            #     'error'
            #     pdb.set_trace()
        assert valid, 'did not find any face'
    # pdb.set_trace()
    face_inds = torch.Tensor(uv_faceids).long()
    face_inds = face_inds.view(uv_map3d.shape[0], uv_map3d.shape[1])
    faces = faces[face_inds, :]

    facesWithBatch = cub_parse.add_bIndex(faces)

    face_verts = torch.stack(
        [
            verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 1]],
            verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 2]],
            verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 3]]
        ],
        dim=2
    )  ## B x H*W x Nverts_per_face x 3

    face_uv_verts = torch.stack(
        [
            uv_verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 1]],
            uv_verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 2]],
            uv_verts[facesWithBatch[:, :, 0], facesWithBatch[:, :, 3]]
        ],
        dim=2
    )  ## B x H*W x Nverts_per_face x 2

    bary_cord = compute_barycentric_coordinates(face_uv_verts, uv_crds)
    points3d = face_verts * bary_cord[..., None]
    points3d = points3d.sum(-2)
    if get_bary:
        return points3d, bary_cord
    else:
        return points3d
