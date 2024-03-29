#!/usr/bin/env python3

import cv2
import rospy
import pandas as pd
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped
import tf
from std_msgs.msg import UInt16, Float32MultiArray
from nmcl_msgs.msg import Omni3D, Omni3DArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import time
import open3d as o3d
from MapObjectTracker import MapObjectTracker
from scipy.spatial.transform import Rotation as R
from data_processing.srv import NoArguments
from VisUtils import *
from JSONUtils import *
from ObjectUtils import *
from DatasetUtils import *
from SIMP import SIMP


class SIMPNode():

	def __init__(self):

		jsonPath = rospy.get_param('jsonPath')
		omniTopic = rospy.get_param('omniTopic')
		gtTopic = rospy.get_param('gtTopic')
		omni3d_sub = Subscriber(omniTopic, Omni3DArray)
		self.dumpPath = rospy.get_param('~dumpPath')
		gt_sub = Subscriber(gtTopic, PoseStamped)
		self.ats = ApproximateTimeSynchronizer([omni3d_sub, gt_sub], queue_size=50000, slop=0.1)
		self.ats.registerCallback(self.callback)
		self.map3d_pub = rospy.Publisher("omni3dmap", MarkerArray, queue_size=1)
		self.map2d_pub = rospy.Publisher("omni2dmap", Image, queue_size=1)
		self.render_pub = rospy.Publisher("render", Image, queue_size=1)
		self.simp = SIMP(jsonPath)
		self.clr = cm.rainbow(np.linspace(0, 1, len(self.simp.categories)))
		self.cnt = 0
		self.bridge = CvBridge()
		self.merge = 0
		self.srv = rospy.Service('/dump_map', NoArguments, self.dumpMap)

		with open(jsonPath) as f:
			self.args = json.load(f)

		self.min_z = self.args['min_z']

		rospy.loginfo("SIMPNode::Ready!")


	def dumpMap(self, request):

		mapObjects = copy.deepcopy(self.simp.mapObjects)
		DumpTrackedObjectsToPickle(self.dumpPath, mapObjects)
		return "dumped"


	def createMarkers(self, mapObjs):

		markers = []

		for obj in mapObjs:

			marker = Marker()
			marker.header.frame_id = "map"
			marker.action = marker.ADD
			marker.type = marker.CUBE;
			marker.pose.position.x = obj.center[0]
			marker.pose.position.y = obj.center[1]
			marker.pose.position.z = obj.center[2] - self.min_z

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
			marker.lifetime = rospy.Duration()

			markers.append(marker)

		return markers




	def callback(self, omni3darray_msg, gt_msg):

		x = gt_msg.pose.position.x
		y = gt_msg.pose.position.y
		z = gt_msg.pose.position.z
		q = gt_msg.pose.orientation
		r = R.from_quat([q.x, q.y, q.z, q.w])
		roll, pitch, yaw = r.as_euler('xyz', degrees=False)
		gt = [x, y, z + self.min_z, yaw]

		self.simp.update(omni3darray_msg.detections, gt)
		# if not (self.merge % 10):
		# 	n = len(self.simp.mapObjects)
		# 	start = time.time()
		# 	self.simp.mapObjects = self.simp.mergeObjects(self.simp.mapObjects, 0.7)
		# 	self.simp.purge()
		# 	m = len(self.simp.mapObjects)
		# 	end = time.time()
		# 	print("merge objects time ", end - start)
			#rospy.loginfo("SIMPNode::Before merge {}, after {}!".format(n, m))

		mapObjs = copy.deepcopy(self.simp.mapObjects)

		markers = self.createMarkers(mapObjs)
		markerArray = MarkerArray()
		markerArray.markers = markers
		#print(markers)

		mid = 0
		for m in markerArray.markers:
			m.id = mid
			self.cnt += 1
			mid += 1
		self.map3d_pub.publish(markerArray) 

		img_msg = self.bridge.cv2_to_imgmsg(self.simp.map2d, encoding="rgb8")
		self.map2d_pub.publish(img_msg)

		tmp = self.simp.debug_render
		if tmp is not None:
			tmp *= 255
			tmp = tmp.astype(np.uint8)
			img_msg = self.bridge.cv2_to_imgmsg(tmp, encoding="passthrough")
			self.render_pub.publish(img_msg)

		self.merge +=1

	



def main():
    

    rospy.init_node('SIMPNode', anonymous=True)
    #rate = rospy.Rate(10)
    simp = SIMPNode()
    rospy.spin()

if __name__ == "__main__":
    main()
