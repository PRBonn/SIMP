#!/usr/bin/env python3

import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import cv2
import json
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import UInt16, Float32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import time
import open3d as o3d
import copy
from DatasetUtils import get_cuboid_verts_faces, convert_3d_box_to_2d, getTrunc2Dbbox
from matplotlib import cm
from scipy.spatial import ConvexHull

sys.path.append(os.path.join(sys.path[0],'/omni3d/'))
print(sys.path)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis



origin = o3d.geometry.TriangleMesh.create_coordinate_frame()



def toLabCoords(center, dim, rot):

    center = np.array([center]).T
    rot = np.array(rot)
    
    R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
    center = (R @ center).T.flatten()
    rot = R @ rot

    return center, dim, rot


def moveToCam(gt, center, dim, rot):

    T = np.eye(4)
    #R = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
    R = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
    T[:3, :3] = R
    T[0, 3] = gt[0]
    T[1, 3] = gt[1]
    T[2, 3] = gt[2]

    center = np.array([center[0], center[1], center[2], 1.0]).T
    rot = np.array(rot)

    center = (T @ center).T.flatten()
    center = center[:3]
    rot = R @ rot

    return center, dim, rot



def parse():

    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    #parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=["MODEL.WEIGHTS", ""],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.opts = ['MODEL.WEIGHTS', '']

   # print("Command Line Args:", args)

    return args
    



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



class Omni3DNode(Node):


    def __init__(self, args):

        super().__init__('Omni3DNode')

        self.declare_parameter('configPath')
        configPath = self.get_parameter('configPath').value
        self.get_logger().info("configPath: %s" % (str(configPath),))
        self.declare_parameter('modelPath')
        modelPath = self.get_parameter('modelPath').value
        self.get_logger().info("modelPath: %s" % (str(modelPath),))
        self.declare_parameter('jsonPath')
        jsonPath = self.get_parameter('jsonPath').value
        self.get_logger().info("jsonPath: %s" % (str(jsonPath),))
        self.declare_parameter('cameraTopics')
        cameraTopics = self.get_parameter('cameraTopics').value
        self.get_logger().info("cameraTopics: %s" % (str(cameraTopics),))
        self.declare_parameter('omni3dTopic')
        omni3dTopic = self.get_parameter('omni3dTopic').value
        self.get_logger().info("omni3dTopic: %s" % (str(omni3dTopic),))

        cameraTopics = ["/camera0/color/image_raw", "/camera1/color/image_raw", "/camera2/color/image_raw", "/camera3/color/image_raw"]

        
        with open(jsonPath) as f:
            self.args = json.load(f)

        args.config_file = configPath
        args.opts =  ['MODEL.WEIGHTS', modelPath]

        cfg = setup(args)
        self.model = build_model(cfg)
        
        #logger.info("Model:\n{}".format(self.model))
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True   )

        self.model.eval()
        self.thres = args.threshold

        #output_dir = cfg.OUTPUT_DIR
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        self.augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
            
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        self.metadata = util.load_json(category_path)
        self.cats = self.metadata['thing_classes']
    
        cam0_sub = Subscriber(self, Image, cameraTopics[0])
        cam1_sub = Subscriber(self, Image, cameraTopics[1])
        cam2_sub = Subscriber(self, Image, cameraTopics[2])
        cam3_sub = Subscriber(self, Image, cameraTopics[3])
        self.camNum = 4
        self.bridge = CvBridge()
        self.detections = []
        self.cnt = 0
		
        cam_h = self.args['camera_z'] + self.args['min_z']
        self.K = np.reshape(np.array([self.args['K']]), (3,3))
        self.cam_poses = np.reshape(np.array([self.args['cam_poses']]), (4,4))
        self.cam_poses[:, 2] += cam_h
        self.angles = self.args['angles']

        self.ats = ApproximateTimeSynchronizer([cam0_sub, cam1_sub, cam2_sub, cam3_sub], queue_size=1, slop=0.05)
        self.ats.registerCallback(self.callbackCombined4)

        self.pred_pub = self.create_publisher(Float32MultiArray, omni3dTopic, queue_size=1)
        self.pred_pub_debug = self.create_publisher(Image, 'omni3d_debug', queue_size=1)
        self.marker_pub = self.create_publisher(MarkerArray, 'omni3dMarkerTopic', queue_size=1)
    
        self.get_logger().info("Omni3DNode::Ready!")



    def createMarker(self, obj_pnts, color):

        pnts = []
        obj_pnts = obj_pnts[:, :2]

        hull = ConvexHull(obj_pnts)
        box = np.zeros((4, 2), np.int32)

        for v in range(4):
            box[v] = obj_pnts[hull.vertices[v]]

        for o in box:
            pnts.append(Point(o[0], o[1], 0.05))

        #obj[:, 2] = 0.05
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.action = marker.ADD
        marker.color.a = 1.0
        marker.color.r = color[2]
        marker.color.g = color[1]
        marker.color.b = color[0]

        marker.type = marker.LINE_STRIP
        marker.points = pnts

        marker.color.a = 1.0
        marker.color.r = color[2]
        marker.color.g = color[1]
        marker.color.b = color[0]
        marker.scale.x = 0.1

        return marker


    def callbackCombined4(self, cam0_msg, cam1_msg, cam2_msg, cam3_msg):

        self.cnt += 1
        if (self.cnt % 5):
            return

        img_msgs = [cam0_msg, cam1_msg, cam2_msg, cam3_msg]
        start = time.time()
        debug_img = self.inferAll(img_msgs)
        end = time.time()
        #print("inference time ", end - start)

        image_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        self.pred_pub_debug.publish(image_msg)
        
       
        msg = Float32MultiArray()
        msg.data = self.detections
        self.pred_pub.publish(msg)




    def inferAll(self, img_msgs):

        batched = []
        self.detections.clear()
        h = img_msgs[0].height
        w = img_msgs[0].width
        debug_img = np.zeros((h, w*self.camNum, 3)).astype(np.uint8)
        imgs = []
        markerArray = MarkerArray()

        for c in range(4):

            im = self.bridge.imgmsg_to_cv2(img_msgs[c], desired_encoding='passthrough')
            image_shape = im.shape[:2]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im)
            aug_input = T.AugInput(im)
            im = aug_input.image
            batched.append({
            'image': torch.as_tensor(np.ascontiguousarray(imgs[c].transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': self.K})

        #start = time.time()
        alldets = self.model(batched)
        #end = time.time()
        #print("inference time ", end - start)

        clr = cm.rainbow(np.linspace(0, 1, len(self.cats )))

        for c in range(4):
            dets = alldets[c]['instances']
            n_det = len(dets)

            meshes = []
            meshes_text = []
            gt_ = self.cam_poses[c]  
            gt = copy.deepcopy(gt_)
            im = imgs[c]

            if n_det > 0:
                for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                        dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                        dets.pred_pose, dets.scores, dets.pred_classes
                    )):

                    # skip
                    if score < self.thres:
                        continue
                    
                    cat = self.cats[cat_idx]

                    bbox3D = center_cam.tolist() + dimensions.tolist()
                    meshes_text.append('{} {:.2f}'.format(cat, score))
                    color = [c/255.0 for c in util.get_color(idx)]
                    box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                    meshes.append(box_mesh)

                    center = center_cam.tolist()
                    dim = dimensions.tolist()
                    R_cam = pose.cpu().detach().numpy()
                    conf = score.cpu().detach().numpy()
                    category = cat_idx.cpu().detach().numpy()

                    center, dim, rot = toLabCoords(center, dim, R_cam)
                    center, dim, rot = moveToCam(gt, center, dim, rot)
                    rot = np.reshape(rot, (3,3))
                    box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
                    verts, faces = get_cuboid_verts_faces(box3d, rot)
                    verts = torch.unsqueeze(verts, dim=0)
                    xyz = np.asarray(verts).reshape((8, 3))
                    xyz = xyz[np.argsort(xyz[:, 2], axis=0)]
                    m_xy = xyz[:4]

                    # marker = self.createMarker(m_xy, clr[category])
                    # markerArray.markers.append(marker)

                    detc = xyz.flatten()
                    detc = np.append(detc, conf)
                    detc = np.append(detc, category)
                    self.detections.extend(detc)

                    xyxy, behind_camera, fully_behind = convert_3d_box_to_2d(self.K, bbox3D, R_cam, clipw=w, cliph=h, XYWH=False, min_z=0.00)
                    bbox2D_proj = xyxy.cpu().detach().numpy().astype(np.int32)

                    if fully_behind:
                        continue
                    
                    bbox2D_trunc = getTrunc2Dbbox(bbox2D_proj, h, w)
                    x1, y1, x2, y2 = bbox2D_trunc
                   
                    color = 255 * clr[category, :3]
                    cv2.rectangle(im,(x1,y1), (x2,y2), color, 2)
                    text1 = "{}".format(self.cats[category])
                    text2 = "{:.2f}".format(conf)
                    cv2.putText(im, text1,(x1+2,y1+40),0,2.0,color, thickness = 3)
                    cv2.putText(im, text2,(x1+2,y1+100),0,2.0,color, thickness = 3)

        

            # if len(meshes) > 0:
            #     im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, self.K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            #     debug_img[: , c*w:(c+1)*w] = im_drawn_rgb
            # else:
            debug_img[: , c*w:(c+1)*w] = im

        # mid = 0
        # for m in markerArray.markers:
        #     m.id = mid
        #     mid += 1
        # self.marker_pub.publish(markerArray) 
           
        


        return debug_img



def main(args=None):
    rclpy.init(args=args)

    omni3d_node = Omni3DNode()

    rclpy.spin(omni3d_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    omni3d_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":

    main()
    
   