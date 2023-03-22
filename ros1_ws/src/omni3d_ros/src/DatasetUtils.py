import numpy as np
import open3d as o3d
import copy
import json 
from Lab3DObject import Lab3DObject
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import torch 
import pandas as pd
from MapObjectTracker import MapObjectTracker
from scipy.spatial import ConvexHull
from typing import Tuple, List
import math
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures.meshes import (
    Meshes,
)
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.renderer import (
    PerspectiveCameras, 
    SoftSilhouetteShader, 
    RasterizationSettings,
    MeshRasterizer
)
from  pytorch3d.structures import *
#from detectron2.structures import BoxMode

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import MeshRenderer as MR

UNIT_CUBE = np.array([
       [-0.5, -0.5, -0.5],
       [ 0.5, -0.5, -0.5],
       [ 0.5,  0.5, -0.5],
       [-0.5,  0.5, -0.5],
       [-0.5, -0.5,  0.5],
       [ 0.5, -0.5,  0.5],
       [ 0.5,  0.5,  0.5],
       [-0.5,  0.5,  0.5]
])


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def get_camera(K, width, height, switch_hands=True, R=None, T=None):

    K = to_float_tensor(K)

    if switch_hands:
        K = K @ torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]).float()

    fx = K[0, 0]
    fy = K[1, 1]
    px = K[0, 2]
    py = K[1, 2]

    if R is None:
        camera = PerspectiveCameras(
            focal_length=((fx, fy),), principal_point=((px, py),), 
            image_size=((height, width),), in_ndc=False
        )
    else:
        camera = PerspectiveCameras(
            focal_length=((fx, fy),), principal_point=((px, py),), 
            image_size=((height, width),), in_ndc=False, R=R, T=T
        )

    return camera

# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def mesh_cuboid(box3d=None, R=None, color=None):

    verts, faces = get_cuboid_verts_faces(box3d, R)
      
    if verts.ndim == 2:
        verts = to_float_tensor(verts).unsqueeze(0)
        faces = to_float_tensor(faces).unsqueeze(0)

    ninstances = len(verts)

    if (isinstance(color, Tuple) or isinstance(color, List)) and len(color) == 3:
        color = torch.tensor(color).view(1, 1, 3).expand(ninstances, 8, 3).float()

    # fixed it for you, Meta
    elif color is not None:
    # pass in a tensor of colors per box
        if color.ndim == 2: 
            color = to_float_tensor(color).unsqueeze(1).expand(ninstances, 8, 3).float()

    device = verts.device

    mesh = Meshes(verts=verts, faces=faces, textures=None if color is None else TexturesVertex(verts_features=color).to(device))

    return mesh


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def render_depth_map(K, box3d, pose, width, height, device=None):
    
    cameras = get_camera(K, width, height)
    renderer = get_basic_renderer(cameras, width, height)

    mesh = mesh_cuboid(box3d, pose)

    if device is not None:
        cameras = cameras.to(device)
        renderer = renderer.to(device)
        mesh = mesh.to(device)

    im_rendered, fragment = renderer(mesh)
    silhouettes = im_rendered[:, :, :, -1] > 0

    zbuf = fragment.zbuf[:, :, :, 0]
    zbuf[zbuf==-1] = math.inf
    depth_map, depth_map_inds = zbuf.min(dim=0)

    return silhouettes, depth_map, depth_map_inds


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def estimate_visibility(K, box3d, pose, width, height, device=None):

    silhouettes, depth_map, depth_map_inds = render_depth_map(K, box3d, pose, width, height, device=device)
    # debugmap = depth_map.cpu().detach().numpy()
    # debugmap = np.clip(debugmap, 0, 255).astype(np.uint8)
    # cv2.imshow("depth_map", debugmap)
    # cv2.waitKey()

    n = silhouettes.shape[0]

    visibilies = []

    for annidx in range(n):

        area = silhouettes[annidx].sum()
        visible = (depth_map_inds[silhouettes[annidx]] == annidx).sum()

        visibilies.append((visible / area).item())

    return visibilies


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def get_basic_renderer(cameras, width, height, use_color=False):

    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0 if use_color else np.log(1. / 1e-4 - 1.) * 1e-4, 
        faces_per_pixel=1, 
        perspective_correct=False,
    )

    if use_color:
        # SoftPhongShader, HardPhongShader, HardFlatShader, SoftGouraudShader
        lights = PointLights(location=[[0.0, 0.0, 0.0]])
        shader = SoftPhongShader(cameras=cameras, lights=lights)
    else:
        shader = SoftSilhouetteShader()

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=shader
    )

    return renderer

# This function is Copyright (c) Meta Platforms, Inc. and affiliates
class MeshRenderer(MR):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments

# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def iou(box_a, box_b, mode='cross', ign_area_b=False):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (array or tensor): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (array or tensor): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        ign_area_b (bool): if true then we ignore area of b. e.g., checking % box a is inside b
    """

    data_type = type(box_a)

    # this mode computes the IoU in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0)
            if not ign_area_b:
                union = union + area_b.unsqueeze(1) - inter

            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) 
            if not ign_area_b:
                union = union + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def intersect(box_a, box_b, mode='cross'):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'cross' or 'list', where cross will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    data_type = type(box_a)

    # this mode computes the intersect in the sense of cross.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'cross':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def estimate_truncation(K, box3d, R, imW, imH):

    # fixed it for you, Meta
    box2d, out_of_bounds, fully_behind =  convert_3d_box_to_2d(K, box3d, R, imW, imH, XYWH=False)
    
    if fully_behind:
        return 1.0

    box2d = box2d.detach().cpu().numpy().tolist()
    #box2d_XYXY = BoxMode.convert(box2d, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    # fixed it for you, Meta
    box2d_XYXY = box2d
    image_box = np.array([0, 0, imW-1, imH-1])

    truncation = 1 - iou(np.array(box2d_XYXY)[np.newaxis], image_box[np.newaxis], ign_area_b=True)

    return truncation.item()



# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def convert_3d_box_to_2d(K, box3d, R=None, clipw=0, cliph=0, XYWH=True, min_z=0.20):
    """
    Converts a 3D box to a 2D box via projection. 
    Args:
        K (np.array): intrinsics matrix 3x3
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
        clipw (int): clip invalid X to the image bounds. Image width is usually used here.
        cliph (int): clip invalid Y to the image bounds. Image height is usually used here.
        XYWH (bool): returns in XYWH if true, otherwise XYXY format. 
        min_z: the threshold for how close a vertex is allowed to be before being
            considered as invalid for projection purposes.
    Returns:
        box2d (flexible): the 2D box results.
        behind_camera (bool): whether the projection has any points behind the camera plane.
        fully_behind (bool): all points are behind the camera plane. 
    """

    # bounds used for vertices behind image plane
    topL_bound = torch.tensor([[0, 0, 0]]).float()
    topR_bound = torch.tensor([[clipw-1, 0, 0]]).float()
    botL_bound = torch.tensor([[0, cliph-1, 0]]).float()
    botR_bound = torch.tensor([[clipw-1, cliph-1, 0]]).float()

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)
    verts2d, verts3d = get_cuboid_verts(K, box3d, R)

    # any boxes behind camera plane?
    verts_behind = verts2d[:, :, 2] <= min_z
    behind_camera = verts_behind.any(1)

    verts_signs = torch.sign(verts3d)

    # check for any boxes projected behind image plane corners
    topL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] < 0)
    topR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] < 0)
    botL = verts_behind & (verts_signs[:, :, 0] < 0) & (verts_signs[:, :, 1] > 0)
    botR = verts_behind & (verts_signs[:, :, 0] > 0) & (verts_signs[:, :, 1] > 0)
    
    # clip values to be in bounds for invalid points
    verts2d[topL] = topL_bound
    verts2d[topR] = topR_bound
    verts2d[botL] = botL_bound
    verts2d[botR] = botR_bound

    x, xi = verts2d[:, :, 0].min(1)
    y, yi = verts2d[:, :, 1].min(1)
    x2, x2i = verts2d[:, :, 0].max(1)
    y2, y2i = verts2d[:, :, 1].max(1)

    fully_behind = verts_behind.all(1)

    width = x2 - x
    height = y2 - y

    if XYWH:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)), dim=1)
    else:
        box2d = torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)

    if squeeze:
        box2d = box2d.squeeze()
        behind_camera = behind_camera.squeeze()
        fully_behind = fully_behind.squeeze()

    return box2d, behind_camera, fully_behind


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def get_cuboid_verts(K, box3d, R=None, view_R=None, view_T=None):

    # make sure types are correct
    K = to_float_tensor(K)
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)

    n = len(box3d)

    if len(K.shape) == 2:
        K = K.unsqueeze(0).repeat([n, 1, 1])

    corners_3d, _ = get_cuboid_verts_faces(box3d, R)
    if view_T is not None:
        corners_3d -= view_T.view(1, 1, 3)
    if view_R is not None:
        corners_3d = (view_R @ corners_3d[0].T).T.unsqueeze(0)
    if view_T is not None:
        corners_3d[:, :, -1] += view_T.view(1, 1, 3)[:, :, -1]*1.25

    # project to 2D
    corners_2d = K @ corners_3d.transpose(1, 2)
    corners_2d[:, :2, :] = corners_2d[:, :2, :] / corners_2d[:, 2, :].unsqueeze(1)
    corners_2d = corners_2d.transpose(1, 2)

    if squeeze:
        corners_3d = corners_3d.squeeze()
        corners_2d = corners_2d.squeeze()

    return corners_2d, corners_3d


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()


# This function is Copyright (c) Meta Platforms, Inc. and affiliates
def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation.
    Args:
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space
        faces: the vertex indices per face
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # make sure types are correct
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    w3d = box3d[:, 3].unsqueeze(1)
    h3d = box3d[:, 4].unsqueeze(1)
    l3d = box3d[:, 5].unsqueeze(1)

    '''
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    '''

    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))

    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
    verts[:, 0, [1, 2, 5, 6]] = l3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
    verts[:, 1, [2, 3, 6, 7]] = h3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
    verts[:, 2, [4, 5, 6, 7]] = w3d / 2

    if R is not None:

        # rotate
        verts = R @ verts
    
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ]).float().unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze()
        faces = faces.squeeze()

    return verts, faces.to(verts.device)



def trans2d(x, y, angle):
    x_ = np.cos(-angle) * x + np.sin(-angle) * y
    y_ = -np.sin(-angle) * x + np.cos(-angle) * y

    return np.array([x_, y_, 1])





# compute the relative transformation between the robot/camera and the 3D object
def getComposedTransform(origin, gt, pose, R):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
    T[0, 3] = gt[0]
    T[1, 3] = gt[1]
    T[2, 3] = gt[2]
    T_inv = np.linalg.inv(T)
    pose_ = pose
    pose_[3] = 1
    
    pose_ = np.array([pose]).T
    new_pose = T_inv @ pose_

    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
    T_inv = np.linalg.inv(T)
    R_cam = np.linalg.inv( T[:3, :3]) @ R

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_cam
    T_rel[0, 3] = new_pose[0]
    T_rel[1, 3] = new_pose[1]
    T_rel[2, 3] = new_pose[2]

    center_cam = T_rel[:3, 3]
    R_cam = T_rel[:3, :3]

    return  R_cam, center_cam


# compute the relative transformation between the robot/camera and the 3D object
def GetPredictionInGTFrame(origin, gt_center, gt_R, pr_center, pr_R):


    T = np.eye(4)
    T[:3, :3] = gt_R
    T[0, 3] = gt_center[0]
    T[1, 3] = gt_center[1]
    T[2, 3] = gt_center[2]
    T_inv = np.linalg.inv(T)
    # pose_ = pr_center
    # pose_[3] = 1
    
    pose_ = np.array([pr_center]).T
    new_pose = T_inv @ pose_

    T = np.eye(4)
    T[:3, :3] = gt_R
    T_inv = np.linalg.inv(T)
    R_cam = np.linalg.inv( T[:3, :3]) @ pr_R

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_cam
    T_rel[0, 3] = new_pose[0]
    T_rel[1, 3] = new_pose[1]
    T_rel[2, 3] = new_pose[2]

    center_cam = T_rel[:3, 3]
    R_cam = T_rel[:3, :3]

    return  R_cam, center_cam


# compute the relative transformation between the robot/camera and the 3D object
def getComposedTransformOmni(origin, gt, pose, R):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, -gt[3], 0))
    T[0, 3] = gt[0]
    T[1, 3] = gt[1]
    T[2, 3] = gt[2]
    T_inv = np.linalg.inv(T)
    pose_ = pose
    pose_[3] = 1
    
    pose_ = np.array([pose]).T
    new_pose = T_inv @ pose_

    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, -gt[3], 0))
    T_inv = np.linalg.inv(T)
    R_cam = np.linalg.inv( T[:3, :3]) @ R

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_cam
    T_rel[0, 3] = new_pose[0]
    T_rel[1, 3] = new_pose[1]
    T_rel[2, 3] = new_pose[2]

    center_cam = T_rel[:3, 3]
    R_cam = T_rel[:3, :3]

    return  R_cam, center_cam


# compute the relative transformation between the 3D object and the camera pose
def getComposedTransformOmniInverse(origin, gt, pose, R):


    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = pose[0]
    T[1, 3] = pose[1]
    T[2, 3] = pose[2]
    T_inv = np.linalg.inv(T)
    gt_ = copy.deepcopy(gt)
    gt_[3] = 1
    
    gt_ = np.array([gt_]).T
    new_gt = T_inv @ gt_

    T = np.eye(4)
    T[:3, :3] = R
    R_cam = np.linalg.inv( T[:3, :3]) @ origin.get_rotation_matrix_from_xyz((0, -gt[3], 0))

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_cam
    T_rel[0, 3] = new_gt[0]
    T_rel[1, 3] = new_gt[1]
    T_rel[2, 3] = new_gt[2]

    center_cam = T_rel[:3, 3]
    R_cam = T_rel[:3, :3]

    return  R_cam, center_cam


# visualizes all objects in the camera frame
def debug3DViz(origin, cubes, colors, args=None, im=None):

    bbs = []
    bbs.append(origin)

    for i, cube in enumerate(cubes):
        xyz = cube
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        bb_t = pcd.get_oriented_bounding_box()
        bb_t.color = colors[i]
        bbs.append(bb_t)

    # if im:
    #     bbs.append(im)
    if args:
        o3d.visualization.draw_geometries(bbs, zoom=args[0], front=args[1], lookat=args[2], up=args[3])
    else: 
        o3d.visualization.draw_geometries(bbs)



def getTrunc2Dbbox(bbox2D_proj, img_h, img_w):

    x, y, x2, y2 = bbox2D_proj

    if x  < 0:
        x = 0
    if x > img_w - 1:
        x = img_w - 1
    if y  < 0:
        y = 0
    if y > img_h - 1:
        y = img_h - 1

    if x2 > img_w - 1:
        x2 = img_w - 1
    if y2 > img_h - 1:
        y2 = img_h - 1

    return np.array([x, y, x2, y2])


def toLabCoords(center, dim, rot):

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()


    center = np.array([center]).T
    rot = np.array(rot)
    
    R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    center = (R @ center).T.flatten()
    rot = R @ rot

    return center, dim, rot




# converts Lab3DObject to a 2D projection on the floor
def obj2UVs(gmap, o, s=1):


    center = o.center
    dim = o.dim
    rot = o.rot
    category = o.category
    conf = o.conf

    center, dim, rot = toLabCoords(center, dim, rot)

    box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
    verts, faces = get_cuboid_verts_faces(box3d, rot)
    verts = np.asarray(verts).reshape((8, 3))
    xyz_t = verts[verts[:, 2].argsort()][:4]
    hull = ConvexHull(xyz_t[:, :2])
    box = np.zeros((4, 2), np.int32)

    for v in range(4):
        #vert = xyz_t[vertIDS[v]]
        vert = xyz_t[hull.vertices[v]]
        uv = gmap.world2map(vert)
        box[v] = uv

    return box

# reads the json file that is produces by segements.ai and parses it to Lab3DObject
def loadObjects(jsonPath):


    with open(jsonPath, 'r') as f:
      data = json.load(f)

    objs = []

    rooms = data['dataset']['samples']
    for r in range(len(rooms)):
        annos = rooms[r]['labels']['ground-truth']['attributes']['annotations']

        for a in annos:
            omo = Lab3DObject(a)
            objs.append(omo)

    return objs






def is_valid3D(gmap, xy):

    uv = gmap.world2map(xy)

    tl = gmap.TopLeft()
    br = gmap.BottomRight()

    if uv[0] < br[0] and uv[1] < br[1] and uv[0] > tl[0] and uv[1] > tl[1]:
        val = gmap.map[uv[1], uv[0]]
        #print(val)
        if val == 0:
            return True

    return False




def is_valid3DSem(gmap, semmap, xy, sid):

    uv = gmap.world2map(xy)

    tl = gmap.TopLeft()
    br = gmap.BottomRight()

    if uv[0] < br[0] and uv[1] < br[1] and uv[0] > tl[0] and uv[1] > tl[1]:
        val = gmap.map[uv[1], uv[0]]
        #print(val)
        if val == 0:
            val2 = semmap[uv[1], uv[0]]
            if val2 == 0 or val2 == sid:
                return True

    return False



def is_valid(gmap, uv):

    tl = gmap.TopLeft()
    br = gmap.BottomRight()

    if uv[0] < br[0] and uv[1] < br[1] and uv[0] > tl[0] and uv[1] > tl[1]:
        val = gmap.map[uv[1], uv[0]]
        #print(val)
        if val == 0:
            return True

    return False


def isTraced(gmap, pose3d, obj3d, semMap, gid):

    pose2d = gmap.world2map(pose3d)
    obj2d = gmap.world2map(obj3d)
    dirc = obj2d - pose2d
    norm = np.linalg.norm(dirc)
    dirc = dirc / norm

    step = 1
    pnt = pose2d.astype("float64")
    pnt += step * dirc

    while(is_valid(gmap, [int(pnt[0]), int(pnt[1])])):
        pix = semMap[int(pnt[1]), int(pnt[0])]
        #print(pix, gid)
        # if pix == gid + 1:
        #     return True
        # elif pix:
        #     return False
        if pix:
            return True
        else:
            pnt += step * dirc


    return False


def isVisibleSem(gmap, semMap, sid, pose3d, obj3d):

    
    p_3d = np.asarray(pose3d[:2])
    o_3d = np.asarray(obj3d[:2])
    dirc = o_3d - p_3d
    norm = np.linalg.norm(dirc)
    dirc = dirc / norm
    eps = 0.07

    step = 0.05
    pnt = p_3d.astype("float64")
    pnt += step * dirc

    while(is_valid3DSem(gmap, semMap, pnt, sid)):
       # print(pnt, o_3d)
       
        if (np.linalg.norm(pnt-o_3d) < step):
        #if pnt[0] == o_3d[0] and pnt[1] == o_3d[1]:
            #print("visibile!")
            return True
        else:
            pnt += step * dirc


    return False


def isVisible(gmap, pose3d, obj3d):

    
    p_3d = np.asarray(pose3d[:2])
    o_3d = np.asarray(obj3d[:2])
    dirc = o_3d - p_3d
    norm = np.linalg.norm(dirc)
    dirc = dirc / norm
    eps = 0.07

    step = 0.05
    pnt = p_3d.astype("float64")
    pnt += step * dirc

    while(is_valid3D(gmap, pnt)):
       # print(pnt, o_3d)
       
        if (np.linalg.norm(pnt-o_3d) < step):
        #if pnt[0] == o_3d[0] and pnt[1] == o_3d[1]:
            #print("visibile!")
            return True
        else:
            pnt += step * dirc


    return False


def rectIntersection(rect1, rect2):

    min_x = max(rect1[0], rect2[0])
    min_y = max(rect1[1], rect2[1])
    max_x = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    max_y = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    if(max_x - min_x) < 0:
        return -1
    if (max_y - min_y) < 0:
        return -1

    return (max_x - min_x) * (max_y - min_y)



def rectUnion(rect1, rect2):

    min_x = min(rect1[0], rect2[0])
    min_y = min(rect1[1], rect2[1])
    max_x = max(rect1[0] + rect1[2], rect2[0] + rect2[2])
    max_y = max(rect1[1] + rect1[3], rect2[1] + rect2[3])

    return (max_x - min_x) * (max_y - min_y)

def IoU(tl1, br1, tl2, br2):

    rect1 = cv2.boundingRect(np.array([tl1, br1]))
    rect2 = cv2.boundingRect(np.array([tl2, br2]))

    interRec = rectIntersection(rect1, rect2)
    unionRec = rectUnion(rect1, rect2)

    if interRec > 0:
        return interRec / unionRec

    return 0





def convertTransform2Omni3D(origin, center_cam, R_cam, dimensions):

    R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    R_inv = np.linalg.inv(R)
    R_cam = R_inv @ R_cam
    center = np.ones((1, 3), dtype=np.float64)
    center[:, :] = center_cam
    center = center.T
    center_cam = R_inv @ center

    return center_cam, R_cam



## old stuff 

def transformMesh(origin, mesh, gt):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, 0, gt[2]))
    T[0, 3] = gt[0]
    T[1, 3] = gt[1]
    T[2, 3] = 0
    mesh_t = copy.deepcopy(mesh).transform(T)

    return mesh_t



def processCube(origin, pcd):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    pcd_t = copy.deepcopy(pcd).transform(T)

    return pcd_t


def rotateCamera(origin, pcd, angle):


    T = np.eye(4)
    T[:3, :3] = origin.get_rotation_matrix_from_xyz((0, 0, angle))
    pcd_t = copy.deepcopy(pcd).transform(T)

    return pcd_t


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta

def wrapTo1Pi(theta):
    while theta < 0:
        theta = theta +  np.pi
    while theta > np.pi:
        theta = theta - np.pi
    return theta

def get_yaw(qz, qw):

    yaw = wrapToPi(2 * np.arctan(qz/qw))
    
    return yaw




def renderMeshes(origin, K, img_w, img_h, mapObjs, gt, device, viz=False):

    clrs = cm.rainbow(np.linspace(0.05, 0.95, len(mapObjs)))
    meshes = []

    cubes = []
    colors = []

    for i, o in enumerate(mapObjs):
        center = o.center
        rot = o.rot
        dim = o.dim
        clr = list(clrs[i][:3])
        
        # get relative pose to camera
        rot, center = getComposedTransformOmni(origin, gt, [center[0], center[1], center[2], 1], rot)

        # convert to omni coordinates
        #center, rot = convertTransform2Omni3D(origin, center, rot, dim)
        center = center.flatten()

        # if viz:
        #     addToDebugViz(cubes, colors, center, dim, rot, [1, 0, 0])

        # create mesh 
        box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
        mesh = mesh_cuboid(box3d, rot, color=clr)
        meshes.append(mesh)

    cameras = get_camera(K, img_w, img_h).to(device)
    renderer = get_basic_renderer(cameras, img_w, img_h, use_color=True).to(device)
    # if viz:
    #     debug3DViz(origin, cubes, colors)
    #print("render done")

    if meshes:
        meshes_scene = join_meshes_as_scene(meshes).cuda()
        meshes_scene.textures = meshes_scene.textures.to(device)

        im_rendered, fragment = renderer(meshes_scene)

    return im_rendered, fragment, meshes, clrs


def getTraceSameClassObjects(objs, category, gmap, gt, semMaps):

    resObjs = []
    sim_objs = [obj for obj in objs if obj.category == category]

    for s, o in enumerate(sim_objs):

        val = isVisible(gmap, [gt[2], -gt[0]], [o.center[2], -o.center[0]])
        #val = isTraced(gmap, gt, simObj.pos, semMaps[category], o.gid)
        if val:
            resObjs.append(o)

    return resObjs



   


def gt2Omni3DCoords(origin, gt):

    R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    R_inv = np.linalg.inv(R)
    # R_gt = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
    # R_gt = R_inv @ R_gt

    center = np.ones((1, 3), dtype=np.float64)
    center[:, :] = gt[:3]
    center = center.T
    center_cam = R_inv @ center
    gt[0] = center_cam[0].item()
    gt[1] = center_cam[1].item()
    gt[2] = center_cam[2].item()

    return gt


def visibilityAnalysis(im_rendered, obj, fragment, color_id, img_h, img_w):

    clr = list(0.5 * color_id[:3])

    lowcolor = (clr[0]-0.01,clr[1]-0.01,clr[2]-0.01)
    highcolor = (clr[0]+0.01,clr[1]+0.01,clr[2]+0.01)
    #print(clr, lowcolor, highcolor)

    # threshold
    thresh = cv2.inRange(im_rendered, lowcolor, highcolor)
    count = np.sum(thresh[np.nonzero(thresh)]) 
    #print(count)
    # cv2.imshow("", im_rendered)
    # cv2.waitKey()
    #if count > 0.005:
    if count:

        zbuf = fragment.zbuf[:, :, :, 0]
        zbuf[zbuf==-1] = math.inf
        depth_map, depth_map_inds = zbuf.min(dim=0)

        depth_map = depth_map.cpu().detach().numpy()
        depth_map[depth_map == np.inf] = -1
        depth_map[depth_map >= 0] = 255
        depth_map[depth_map < 0] = 0
        # cv2.imshow("", depth_map)
        # cv2.waitKey()
        #print(np.unique(depth_map))
        count_single = np.sum(depth_map[np.nonzero(depth_map)]) 

        # if the size of the object is below some TH, consider it too small to see
        if count_single / (255 * img_h * img_w) < 0.004:
            return False
        #print("ratio ", count / count_single)

        # if too much of the object is occluded by other objects, consider it unseen
        if (count / count_single) > 0.66:
            return True

    return False

            


def getDist2Cuboid(obj, gt):

    center = obj.center
    dim = obj.dim
    box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
    verts, faces = get_cuboid_verts_faces(box3d, obj.rot)
    xyz = np.asarray(verts).reshape((8, 3))
    xyz = xyz[xyz[:, 1].argsort()][:4]
    xy1 = [xyz[0][2], xyz[0][0]]
    xy2 = [xyz[1][2], xyz[1][0]]
    xy3 = [xyz[2][2], xyz[2][0]]
    xy4 = [xyz[3][2], xyz[3][0]]
      
    p1, p2, p3, p4 = map(Point, [xy1, xy2, xy3, xy4])
    poly = Polygon(p1, p2, p3, p4)
    shortestDistance = N(poly.distance(Point(gt[2], gt[0])))
      
    return shortestDistance