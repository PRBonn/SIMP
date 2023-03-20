#!/usr/bin/env python3

import cv2
import json
import rospy
import pandas as pd
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
from scipy.spatial.transform import Rotation as R


class GTNode():

	def __init__(self):

		csvPath = rospy.get_param('~csvPath')
		gtTopic = rospy.get_param('gtTopic')
		self.df = pd.read_csv(csvPath)
		jsonPath = rospy.get_param('jsonPath')
		with open(jsonPath) as f:
			self.args = json.load(f)
		self.camera_z = self.args['camera_z']
		cameraTopic = rospy.get_param('cameraTopic')


		self.sub = rospy.Subscriber(cameraTopic, Image, self.callback)

		self.pose_pub = rospy.Publisher(gtTopic, PoseStamped, queue_size=1)
		self.tf_pub = tf.TransformBroadcaster()

		rospy.loginfo("GTNode::Ready!")


	def callback(self, cam0_msg):

		t_img = cam0_msg.header.stamp.to_nsec()
		row = self.df.iloc[(self.df['t']-t_img).abs().argsort()[:1]]
		x = row['gt_x'].to_numpy()[0]
		y = row['gt_y'].to_numpy()[0]
		z = self.camera_z
		yaw = row['gt_yaw'].to_numpy()[0] 
		#print(x, y, yaw, t_img)

		quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
		pose_msg = PoseStamped()
		pose_msg.pose.orientation.x = quaternion[0]
		pose_msg.pose.orientation.y = quaternion[1]
		pose_msg.pose.orientation.z = quaternion[2]
		pose_msg.pose.orientation.w = quaternion[3]
		pose_msg.pose.position.x = x
		pose_msg.pose.position.y = y
		pose_msg.pose.position.z = z
		pose_msg.header.stamp = cam0_msg.header.stamp
		pose_msg.header.frame_id = "map"

		self.pose_pub.publish(pose_msg)




def main():
    

    rospy.init_node('GTNode', anonymous=True)
    #rate = rospy.Rate(10)
    gt = GTNode()
    rospy.spin()

if __name__ == "__main__":

    main()
