import torch
import torch.nn as nn
import numpy as np
from . import geom_utils
from . import misc as misc_utils


class UVTo3D(nn.Module):
    def __init__(self, mean_shape):
        super(UVTo3D, self).__init__()
        self.mean_shape = mean_shape
        # self.verts_3d = mean_shape['verts']

        self.face_inds = mean_shape['face_inds']
        self.faces = mean_shape['faces']
        self.verts_uv = mean_shape['uv_verts']
        # pdb.set_trace()

    def remove_component_perpendicular_to_face(self, verts, points):
        vertA = verts[..., 0, :]
        vertB = verts[..., 1, :]
        vertC = verts[..., 2, :]

        AB = vertB - vertA
        AC = vertC - vertA
        AP = points - vertA

        perp_direction = torch.cross(AB, AC, dim=-1)
        perp_direction = torch.nn.functional.normalize(
            perp_direction, dim=-1, p=2
        )
        vperp = AP - perp_direction * ((perp_direction * AP).sum(-1)[..., None])
        # pdb.set_trace()
        new_points = vperp + vertA
        # self.check_point_inside_triangle(verts, points)
        return new_points

    def check_point_inside_triangle(self, triangle, points):
        matrixM = triangle.data.cpu().numpy().reshape(
            -1, triangle.shape[2], triangle.shape[3]
        )
        matrixM_inv = np.linalg.inv(matrixM)
        points_np = points.data.cpu().numpy().reshape(-1, points.shape[2], 1)
        # pdb.set_trace()

        return

    def compute_barycentric_coordinates(self, uv_verts, uv_points):

        verts = geom_utils.convert_uv_to_3d_coordinates(uv_verts)
        points = geom_utils.convert_uv_to_3d_coordinates(uv_points)

        points = self.remove_component_perpendicular_to_face(verts, points)
        vertA = verts[..., 0, :]
        vertB = verts[..., 1, :]
        vertC = verts[..., 2, :]

        AB = vertB - vertA
        AC = vertC - vertA
        BC = vertC - vertB
        AP = points - vertA
        BP = points - vertB
        CP = points - vertC
        # pdb.set_trace()

        areaBAC = torch.norm(torch.cross(AB, AC, dim=-1), dim=-1)
        areaBAP = torch.norm(torch.cross(AB, AP, dim=-1), dim=-1)
        areaCAP = torch.norm(torch.cross(AC, AP, dim=-1), dim=-1)
        areaCBP = torch.norm(torch.cross(BC, BP, dim=-1), dim=-1)
        # pdb.set_trace()
        w = areaBAP / areaBAC
        v = areaCAP / areaBAC
        u = areaCBP / areaBAC
        barycentric_coordinates = torch.stack([u, v, w], dim=-1)
        barycentric_coordinates = torch.nn.functional.normalize(
            barycentric_coordinates, p=1, dim=-1
        )
        return barycentric_coordinates

    '''
    uv : UV coordinates to convert to 3D
    verts: vertices on the mesh.
    TODO. Need to make verts of Bsize to support batch passes.
    '''

    def forward(self, verts, uv):
        mean_shape = self.mean_shape
        device_id = verts.get_device()
        B, HW = uv.shape[0], uv.shape[1]
        uv_verts = self.verts_uv.unsqueeze(0) * 1
        uv_verts = uv_verts.to(device_id).repeat(B, 1, 1)

        uv_map_size = torch.Tensor(
            [
                mean_shape['uv_map'].shape[1] - 1,
                mean_shape['uv_map'].shape[0] - 1
            ]
        ).view(1, 2)
        uv_map_size = uv_map_size.type(uv.type()).to(device_id)
        # pdb.set_trace()
        uv_inds = (uv_map_size * uv).round().long().detach()

        if torch.max(uv_inds) > uv_map_size[
            0, 0].item() or torch.min(uv_inds) < 0:
            print('Error in indexing')
            pdb.set_trace()

        ## remember this. swaped on purpose. U is along the columns, V is along the rows\
        face_inds = (self.face_inds * 1).to(device_id)
        face_inds = face_inds[uv_inds[..., 1], uv_inds[..., 0]]
        faces = (self.faces * 1).to(device_id)
        faces = faces[face_inds, :]
        # pdb.set_trace()

        facesWithBatch = misc_utils.add_bIndex(faces)

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

        bary_cord = self.compute_barycentric_coordinates(face_uv_verts, uv)
        points3d = face_verts * bary_cord[..., None]
        points3d = points3d.sum(-2)
        return points3d
