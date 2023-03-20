/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: OptiNode.cpp                                                          #
# ##############################################################################
**/

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>

#include "Utils.h"
#include "OptiTrack.h"
#include "RosUtils.h"

class OptiNode
{
	public:

		OptiNode()
		{
			ros::NodeHandle nh;

			if (!ros::param::has("dataFolder"))
			{
				std::cout << "data folder not found!" << std::endl;
			}


			std::string dataFolder;
			std::string trackingTopic;
			nh.getParam("dataFolder", dataFolder);
			nh.getParam("trackingTopic", trackingTopic);
			nh.getParam("mapTopic", mapTopic);


			op = std::make_shared<OptiTrack>(OptiTrack(dataFolder));

			posePub = nh.advertise<geometry_msgs::PoseStamped>("/OptiPose", 10);
			poseSub = nh.subscribe(trackingTopic, 10, &OptiNode::Callback, this);


		}


		void Callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
		{
		  //ROS_INFO("I heard: [%s]", msg->data.c_str());

			float x = msg->pose.position.x;
			float y = msg->pose.position.y;
			float qz = msg->pose.orientation.z;
			float qw = msg->pose.orientation.w;
			Eigen::Vector3f p_trans = op->OptiTrack2World(Eigen::Vector3f(x, y, GetYaw(qz, qw)));


			geometry_msgs::PoseStamped poseStamped = Pose2D2PoseMsg(p_trans);
			poseStamped.header.frame_id = mapTopic;
			poseStamped.header.stamp = msg->header.stamp;
			posePub.publish(poseStamped);
		}

	private:
		ros::Publisher posePub;
		ros::Subscriber poseSub;
		std::shared_ptr<OptiTrack> op;
		std::string mapTopic;
};



int main(int argc, char** argv)
{
	ros::init(argc, argv, "OptiNode");

	OptiNode on = OptiNode();
	ros::spin();


	return 0;
}