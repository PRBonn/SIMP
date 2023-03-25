#!/usr/bin/env python3


import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import get_cmap
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import yaml
import os
import sys
from pathlib import Path
from GMAP import GMAP

class SemanticMapNode(Node):

	def __init__(self)->None:
		super().__init__('SemanticMapNode')

		self.declare_parameter('dataset')
		data = self.get_parameter('dataset').value
		self.get_logger().info("dataset: %s" % (data,))

		self.declare_parameter('dataFolder')
		dataFolder = self.get_parameter('dataFolder').value
		self.get_logger().info("dataFolder: %s" % (dataFolder,))
		if os.path.exists(dataFolder + "0/"):
			dataFolder = dataFolder + "0/"

		semFolder = dataFolder + "SemMaps/"

		self.declare_parameter('mapName')
		mapName = self.get_parameter('mapName').value
		self.get_logger().info("mapName: %s" % (mapName,))

		self.declare_parameter('markerTopic')
		markerTopic = self.get_parameter('markerTopic').value
		self.get_logger().info("markerTopic: %s" % (markerTopic,))

		publisher = self.create_publisher(MarkerArray, markerTopic, 10)

		yamlfile = dataFolder + mapName + ".yaml"
		with open(yamlfile, 'r') as stream:
		    map_loaded = yaml.safe_load(stream)

		gridmap = cv2.imread(dataFolder + map_loaded['image'])
		resolution = map_loaded['resolution']
		origin = map_loaded['origin']
		gmap = GMAP(gridmap, resolution, origin)

		with open(data, 'r') as stream:
		    data_loaded = yaml.safe_load(stream)

		semclasses = data_loaded['names']
		classCnt = len(semclasses)
		clr = cm.rainbow(np.linspace(0, 1, len(semclasses)))

		objects = []

		for c in range(classCnt):

			perclass = []

			objectMap = cv2.imread(semFolder + semclasses[c] + ".png", 0)
			ret, thresh = cv2.threshold(objectMap, 127, 255, 0)
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				pnts = []
				for p in cnt:
					xy = gmap.map2world(p[0])
					p = Point()
					p.x = xy[0]
					p.y = xy[1]
					p.z = 0.05
					pnts.append(p)

				pnts.append(pnts[0])
				perclass.append(pnts)

			objects.append(perclass)

		count = 0
		MARKERS_MAX = 100
		

		while rclpy.ok():

			markerArray = MarkerArray()

			for c in range(classCnt):

				color = clr[c]

				txtmarker = Marker()
				txtmarker.header.frame_id = "map"
				txtmarker.type = txtmarker.TEXT_VIEW_FACING
				txtmarker.action = txtmarker.ADD
				txtmarker.text = semclasses[c]
				txtmarker.scale.x = 1.0
				txtmarker.scale.y = 1.0
				txtmarker.scale.z = 1.0
				txtmarker.color.a = 1.0
				txtmarker.color.r = color[2]
				txtmarker.color.g = color[1]
				txtmarker.color.b = color[0]
				txtmarker.pose.orientation.w = 1.0
				txtmarker.pose.position.x = float(-18) 
				txtmarker.pose.position.y =  8 - c * 1.2 
				txtmarker.pose.position.z = 0.01
				markerArray.markers.append(txtmarker)


				z = 0.01
				if semclasses[c] in ["door", "cardboard", "people", "chair"]:
					continue
				if semclasses[c] in ["drawers", "oven"]:
					z = 0.02

				for o in range(len(objects[c])):

					obj = objects[c][o]
					marker = Marker()
					marker.header.frame_id = "map"
					marker.action = marker.ADD
					marker.color.a = 1.0
					marker.color.r = color[2]
					marker.color.g = color[1]
					marker.color.b = color[0]

					marker.type = marker.LINE_STRIP
					marker.points = obj

					marker.color.a = 1.0
					marker.color.r = color[2]
					marker.color.g = color[1]
					marker.color.b = color[0]
					marker.scale.x = 0.1

					markerArray.markers.append(marker)

			# Renumber the marker IDs
			id = 0
			for m in markerArray.markers:
				m.id = id
				id += 1

			# Publish the MarkerArray
			publisher.publish(markerArray)

			#count += 1

			#rospy.sleep(0.01)



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = SemanticMapNode()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()