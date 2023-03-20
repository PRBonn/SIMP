#!/usr/bin/env python3

from GMAP import GMAP
import cv2
import yaml
import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import get_cmap
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

class SemanticMapNode():

	def __init__(self)->None:

		data = rospy.get_param('dataset')
		dataFolder = rospy.get_param('dataFolder') 
		#mapName = rospy.get_param('mapName')
		mapName = "Map"
		semFolder = dataFolder + "SemMaps/"
		markerTopic = rospy.get_param('markerTopic')


		publisher = rospy.Publisher(markerTopic, MarkerArray)

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

		# colors = cm.rainbow(np.linspace(0, 1, 20))
		# clrarr = np.linspace(0.0, 20.0, num=14, endpoint=False).astype("int")
		# clr = [colors[i] for i in range(len(clrarr))]

		# colors = get_cmap('tab20').colors
		# clrarr = np.linspace(0.0, 20.0, num=14, endpoint=False).astype("int")
		# print(clrarr)
		# clr = [colors[i] for i in range(len(clrarr))]

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
					pnts.append(Point(xy[0], xy[1], 0.05))

				pnts.append(pnts[0])
				perclass.append(pnts)

			objects.append(perclass)

		count = 0
		MARKERS_MAX = 100
		

		while not rospy.is_shutdown():

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
				txtmarker.pose.position.x = -18 
				txtmarker.pose.position.y =  4 - c * 1.2 
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

			rospy.sleep(0.01)







if __name__ == "__main__":


    rospy.init_node('SemanticMapNode', anonymous=True)
    #rate = rospy.Rate(10)
    posn = SemanticMapNode()
    rospy.spin()