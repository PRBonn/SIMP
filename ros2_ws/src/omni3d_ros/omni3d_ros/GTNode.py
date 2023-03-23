#!/usr/bin/env python3

import cv2
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf_transformations
from rclpy.time import Time

class GTNode(Node):

	def __init__(self):
		super().__init__('GTNode')

		self.declare_parameter('csvPath')
		csvPath = self.get_parameter('csvPath').value
		self.get_logger().info("csvPath: %s" % (str(csvPath),))
		self.declare_parameter('gtTopic')
		gtTopic = self.get_parameter('gtTopic').value
		self.get_logger().info("gtTopic: %s" % (str(gtTopic),))
		self.declare_parameter('jsonPath')
		jsonPath = self.get_parameter('jsonPath').value
		self.get_logger().info("jsonPath: %s" % (str(jsonPath),))
		self.declare_parameter('cameraTopic')
		cameraTopic = self.get_parameter('cameraTopic').value
		self.get_logger().info("cameraTopic: %s" % (str(cameraTopic),))

		self.df = pd.read_csv(csvPath)
		with open(jsonPath) as f:
			self.args = json.load(f)
		self.camera_z = self.args['camera_z']

		self.sub = self.create_subscription(Image, cameraTopic, self.callback, 1)
		self.pose_pub = self.create_publisher(PoseStamped, gtTopic, 1)

		self.get_logger().info("GTNode::Ready!")


	def callback(self, cam0_msg):

		t_img = Time.from_msg(cam0_msg.header.stamp).nanoseconds
		row = self.df.iloc[(self.df['t']-t_img).abs().argsort()[:1]]
		x = row['gt_x'].to_numpy()[0]
		y = row['gt_y'].to_numpy()[0]
		z = self.camera_z
		yaw = row['gt_yaw'].to_numpy()[0] 

		quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
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




def main(args=None):
    

   # rospy.init_node('GTNode', anonymous=True)
    #rate = rospy.Rate(10)
    #gt = GTNode()
    #rospy.spin()

	rclpy.init(args=args)

	gt_node = GTNode()

	rclpy.spin(gt_node)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	gt_node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":

    main()
