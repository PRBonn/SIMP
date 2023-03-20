#!/usr/bin/env python3

import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import cv2
import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import UInt16, Float32MultiArray
from nmcl_msgs.msg import Omni3D, Omni3DArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import time
import open3d as o3d
import copy
from DatasetUtils import get_cuboid_verts_faces, convert_3d_box_to_2d, getTrunc2Dbbox
from matplotlib import cm
from scipy.spatial import ConvexHull
from MapObjectTracker import MapObjectTracker
from scipy.spatial.transform import Rotation as R
from ObjectUtils import *

origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

all_classes = ['sink', 'door', 'oven', 'board', 'table', 'box', 'potted plant', 'drawers', 'sofa', 'cabinet', 'chair', 'fire extinguisher', 'person', 'desk']
camere_z = 0.63
min_z = -1.55708286


class BroadcastMapDNode():


    def __init__(self):

        self.marker_pub = rospy.Publisher("omni3dMarkerTopic", MarkerArray,  queue_size=10)
       
        self.clr = cm.rainbow(np.linspace(0, 1, 14))

        rospy.loginfo("BroadcastMapDNode::Ready!")

        mapObjects = LoadTrackedObjectsFromPickle("/home/nickybones/Code/OmniNMCL/ros1_ws/src/omni3d_ros/", "mapObjects_ICP2.pickle")
        markers = self.createMarkers(mapObjects)

        while not rospy.is_shutdown():

            markerArray = MarkerArray()
            markerArray.markers = markers
            # Renumber the marker IDs
            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            # Publish the MarkerArray
            self.marker_pub .publish(markerArray)

            rospy.sleep(0.01)



    def createMarkers(self, mapObjs):

        markers = []

        for obj in mapObjs:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.action = marker.ADD
            marker.type = marker.CUBE;
            marker.pose.position.x = obj.center[0]
            marker.pose.position.y = obj.center[1]
            marker.pose.position.z = obj.center[2] - min_z

            r = R.from_matrix(obj.rot)
            q = r.as_quat()
            color = self.clr[obj.category]

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = obj.dim[2]
            marker.scale.y = obj.dim[1]
            marker.scale.z = obj.dim[0]
            marker.color.a = 0.8
            marker.color.r = color[2]
            marker.color.g = color[1]
            marker.color.b = color[0]
            marker.lifetime = rospy.Duration(3)

            markers.append(marker)

        return markers


    

def main():
    

    rospy.init_node('BroadcastMapDNode', anonymous=True)
    omn = BroadcastMapDNode()
    rospy.spin()

if __name__ == "__main__":

    main()
    
    