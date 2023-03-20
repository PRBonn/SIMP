/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ConfigNMCLNode.cpp                                                   #
# ##############################################################################
**/
 
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2/LinearMath/Quaternion.h> 
#include <tf2_ros/transform_listener.h>  
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>   
 
#include <mutex> 
#include <sstream>  

#include "Camera.h"
#include "Utils.h"
#include "ReNMCL.h"
#include "RosUtils.h"

#include <boost/archive/text_iarchive.hpp>
#include <nlohmann/json.hpp>
#include "NMCLFactory.h"
#include "Semantic3DData.h" 
#include "std_msgs/Float32MultiArray.h"         
  
#define DEBUG  
    

class ConfigNMCLNode     
{
public:
	ConfigNMCLNode()       
	{
		
		ros::NodeHandle nh;
		int numParticles;

		if (!ros::param::has("dataFolder"))
		{
			ROS_FATAL_STREAM("Data folder not found!");
		} 

		std::string dataFolder; 
		std::string odomTopic;
		std::vector<double> odomNoise;  
		std::string poseTopic;
		std::string nmclconfig;
		std::string omni3dTopic;
		std::string particleTopic;
		bool useLidar;
 
		nh.getParam("dataFolder", dataFolder);		
		nh.getParam("odomTopic", odomTopic); 
		nh.getParam("mapTopic", o_mapTopic);
		nh.getParam("odomNoise", odomNoise);  
		nh.getParam("triggerDist", o_triggerDist); 
		nh.getParam("triggerAngle", o_triggerAngle);
		nh.getParam("poseTopic", poseTopic); 
		nh.getParam("nmclconfig", nmclconfig);    
		nh.getParam("baseLinkTF", o_baseLinkTF);
		nh.getParam("omni3dTopic", omni3dTopic);
		nh.getParam("particleTopic", particleTopic);
		


		srand48(21);         
		o_mtx = new std::mutex();      
		o_renmcl = NMCLFactory::Create(dataFolder + nmclconfig);  
		
		nav_msgs::OdometryConstPtr odom = ros::topic::waitForMessage<nav_msgs::Odometry>(odomTopic, ros::Duration(60)); 
		o_prevPose = OdomMsg2Pose2D(odom);  
		o_prevTriggerPose = o_prevPose;

		o_odomSub = nh.subscribe(odomTopic, 10, &ConfigNMCLNode::motionCallback, this);   
		o_odomNoise = Eigen::Vector3f(odomNoise[0], odomNoise[1], odomNoise[2]); 		 
		o_posePub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(poseTopic, 10); 
		o_particlePub = nh.advertise<geometry_msgs::PoseArray>(particleTopic, 10);
		o_omniSub = nh.subscribe(omni3dTopic, 1, &ConfigNMCLNode::omniCallback, this);


		ROS_INFO_STREAM("Engine running!");            
 
	}    

	
   

 
	void motionCallback(const nav_msgs::OdometryConstPtr& odom)
	{
		Eigen::Vector3f currPose = OdomMsg2Pose2D(odom);

		Eigen::Vector3f delta = currPose - o_prevTriggerPose;


		Eigen::Vector3f u = o_renmcl->Backward(o_prevPose, currPose);  

		std::vector<Eigen::Vector3f> command{u};
		o_mtx->lock();
		o_renmcl->Predict(command, o_odomWeights, o_odomNoise);
		SetStatistics stas = o_renmcl->Stats();
		o_mtx->unlock(); 

		o_prevPose = currPose; 
		//std::cout << o_prevPose(0) << ", " << o_prevPose(1) << ", " <<  o_prevPose(2) << std::endl;

		//ROS_INFO_STREAM("IN MOTION " << std::to_string(currPose(0)) << ", " << std::to_string(currPose(1)) << ", " <<  std::to_string(currPose(2)) << "\n");
		//ROS_INFO_STREAM("IN MOTION\n");
		Eigen::Matrix3d cov = stas.Cov();
		Eigen::Vector3d pred = stas.Mean();   
		o_pred = pred;   
		o_cov = cov;

		// geometry_msgs::PoseWithCovarianceStamped poseStamped = Pred2PoseWithCov(o_pred, o_cov);
		// poseStamped.header.frame_id = o_mapTopic;
		// poseStamped.header.stamp = ros::Time::now(); 
		// o_posePub.publish(poseStamped); 

		// tf::Transform transform;
	    // transform.setOrigin( tf::Vector3(o_pred(0), o_pred(1), 0.0) );
		// tf::Quaternion q;
		// q.setRPY(0, 0, o_pred(2));
		// transform.setRotation(q);
		// o_tfBroadcast.sendTransform(tf::StampedTransform(transform.inverse(), ros::Time::now(), o_baseLinkTF, o_mapTopic));		

		if((((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > o_triggerDist) || (fabs(delta(2)) > o_triggerAngle)) || o_first)
		{
			//ROS_INFO_STREAM("NMCL first step!");
			o_first = false;
			o_prevTriggerPose = currPose;  
			o_step = true; 
			//ROS_INFO_STREAM("EXECUTE MOTION\n");
		}
	}


	void omniCallback(const std_msgs::Float32MultiArray::ConstPtr& obj3dMsg)
 	{
 		std::vector<int> labels;
		std::vector<std::vector<Eigen::Vector3f>> vertices;
		std::vector<float> confidences;

		//ROS_INFO_STREAM("omniCallback!"); 
 		std::vector<float> detections = obj3dMsg->data;
 		int n = int(detections.size() / 26);     
 		for(int i = 0; i < n ; ++i) 
    	{
    		std::vector<float> detection = {detections.begin() + i * 26, detections.begin() + (i + 1) * 26}; 
    		float confidence = detection[24];
    		int category = int(detection[25]);
    		std::vector<Eigen::Vector3f> cube = std::vector<Eigen::Vector3f>(8);
			for(int v = 0; v < 8; ++v)
			{
				Eigen::Vector3f vert = Eigen::Vector3f(detection[v * 3], detection[v * 3 + 1], detection[v * 3 + 2]);
				cube[v] = vert;
			}

            vertices.push_back(cube);
            labels.push_back(category);
			confidences.push_back(confidence);
    	}
    	if (labels.size())
		{
			//ROS_INFO_STREAM("omniCallback infer!");
			Semantic3DData data = Semantic3DData(labels, vertices, confidences);
			o_mtx->lock();
			o_renmcl->CorrectOmni(std::make_shared<Semantic3DData>(data));
			SetStatistics stas = o_renmcl->Stats();
			std::vector<Particle> particles = o_renmcl->Particles();    
			o_mtx->unlock();  
			Eigen::Matrix3d cov = stas.Cov();
			Eigen::Vector3d pred = stas.Mean(); 
			if(pred.array().isNaN().any() || cov.array().isNaN().any() || cov.array().isInf().any())
			{ 
				ROS_ERROR_STREAM("NMCL fails to Localize!\n");
				o_renmcl->Recover();
			}
			else
			{
				o_pred = pred;   
				publishPose(pred, cov, particles);
			}
			//o_stepOmni = false;
		}


	}

	void publishPose(Eigen::Vector3d pred, Eigen::Matrix3d cov, const std::vector<Particle>& particles)
	{
		geometry_msgs::PoseWithCovarianceStamped poseStamped = Pred2PoseWithCov(pred, cov);
		poseStamped.header.frame_id = o_mapTopic;
		poseStamped.header.stamp = ros::Time::now(); 
		o_posePub.publish(poseStamped); 

		geometry_msgs::PoseArray posearray;
		posearray.header.stamp = ros::Time::now();  
		posearray.header.frame_id = o_mapTopic;
		posearray.poses = std::vector<geometry_msgs::Pose>(particles.size());

		for (int i = 0; i < particles.size(); ++i)
		{
			geometry_msgs::Pose p;
			p.position.x = particles[i].pose(0); 
			p.position.y = particles[i].pose(1);
			p.position.z = 0.1; 
			tf2::Quaternion q;
			q.setRPY( 0, 0, particles[i].pose(2)); 
			p.orientation.x = q[0];
			p.orientation.y = q[1];
			p.orientation.z = q[2];
			p.orientation.w = q[3];

			posearray.poses[i] = p;
		}

		o_particlePub.publish(posearray);

		tf::Transform transform;
	    transform.setOrigin( tf::Vector3(pred(0), pred(1), 0.0) );
		tf::Quaternion q;
		q.setRPY(0, 0, pred(2));
		transform.setRotation(q);
		o_tfBroadcast.sendTransform(tf::StampedTransform(transform.inverse(), ros::Time::now(), o_baseLinkTF, o_mapTopic));		
	}

private:

	tf::TransformBroadcaster o_tfBroadcast;
	tf::TransformBroadcaster o_tfMaskBroadcast;
	ros::Publisher o_posePub;
	ros::Publisher o_particlePub;
	ros::Subscriber o_odomSub;
	ros::Subscriber o_omniSub;

	std::vector<float> o_odomWeights = {1.0};
	Eigen::Vector3f o_prevPose = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f o_prevTriggerPose = Eigen::Vector3f(0, 0, 0);

	Eigen::Vector3f o_odomNoise = Eigen::Vector3f(0.02, 0.02, 0.02);
	Eigen::Vector3d o_pred = Eigen::Vector3d(0, 0, 0);
	Eigen::Matrix3d o_cov;
	std::string o_maskTopic;
	
	std::shared_ptr<ReNMCL> o_renmcl;

	std::string o_mapTopic;
	std::string o_baseLinkTF;
	bool o_first = true;
	bool o_step = false;
	std::mutex* o_mtx;
	float o_triggerDist = 0.05;
	float o_triggerAngle = 0.05;
	int o_cnt = 0;
};



int main(int argc, char** argv)
{
	ros::init(argc, argv, "ConfigNMCLNode");
	ConfigNMCLNode nmcl = ConfigNMCLNode();
	ros::spin();
	

	return 0;
}
