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
 

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
 
//#include <tf/transform_broadcaster.h>   
 
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
  
#define DEBUG  
    

class ConfigNMCLNode   : public rclcpp::Node  
{
public:
	ConfigNMCLNode() : Node("ConfigNMCLNode")     
	{
		

		std::string dataFolder; 
		std::string odomTopic;
		std::string poseTopic;
		std::string nmclconfig;
		std::string omni3dTopic;
		std::string particleTopic;

		this->declare_parameter("dataFolder");
		this->declare_parameter("odomTopic");
		this->declare_parameter("mapTopic");
		this->declare_parameter("odomNoise");
		this->declare_parameter("triggerDist");
		this->declare_parameter("triggerAngle");
		this->declare_parameter("poseTopic");
		this->declare_parameter("nmclconfig");
		this->declare_parameter("baseLinkTF");
		this->declare_parameter("omni3dTopic");
		this->declare_parameter("particleTopic");

		this->get_parameter("dataFolder", dataFolder);
		RCLCPP_INFO(this->get_logger(), "dataFolder %s", dataFolder.c_str());
		this->get_parameter("odomTopic", odomTopic);
		RCLCPP_INFO(this->get_logger(), "odomTopic %s", odomTopic.c_str());
		this->get_parameter("mapTopic", o_mapTopic);
		RCLCPP_INFO(this->get_logger(), "mapTopic %s", o_mapTopic.c_str());
		this->get_parameter("poseTopic", poseTopic);
		RCLCPP_INFO(this->get_logger(), "poseTopic %s", poseTopic.c_str());
		this->get_parameter("nmclconfig", nmclconfig);
		RCLCPP_INFO(this->get_logger(), "nmclconfig %s", nmclconfig.c_str());
		this->get_parameter("triggerDist", o_triggerDist);
		RCLCPP_INFO(this->get_logger(), "triggerDist %f", o_triggerDist);
		this->get_parameter("triggerAngle", o_triggerAngle);
		RCLCPP_INFO(this->get_logger(), "triggerAngle %f", o_triggerAngle);
		this->get_parameter("baseLinkTF", o_baseLinkTF);
		RCLCPP_INFO(this->get_logger(), "baseLinkTF %s", o_baseLinkTF.c_str());
		this->get_parameter("omni3dTopic", omni3dTopic);
		RCLCPP_INFO(this->get_logger(), "omni3dTopic %s", omni3dTopic.c_str());
		this->get_parameter("particleTopic", particleTopic);
		RCLCPP_INFO(this->get_logger(), "particleTopic %s", particleTopic.c_str());


		rclcpp::Parameter dblArrParam =this->get_parameter("odomNoise");
		std::vector<double> odomNoise = dblArrParam.as_double_array();
		RCLCPP_INFO(this->get_logger(), "odomNoise %f %f %f", odomNoise[0], odomNoise[1], odomNoise[2]);
		o_odomNoise = Eigen::Vector3f(odomNoise[0], odomNoise[1], odomNoise[2]);

		srand48(21); 
		o_mtx = new std::mutex();  
    o_renmcl = NMCLFactory::Create(dataFolder + nmclconfig);  

		rclcpp::QoS qos(10);
  	o_posePub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(poseTopic, 10);
  	o_particlePub = this->create_publisher<geometry_msgs::msg::PoseArray>(particleTopic, 10);

  	o_odomSub = create_subscription<nav_msgs::msg::Odometry>(odomTopic, qos, std::bind(&ConfigNMCLNode::motionCallback, this, std::placeholders::_1));
  	o_omniSub = create_subscription<std_msgs::msg::Float32MultiArray>(omni3dTopic, qos, std::bind(&ConfigNMCLNode::omniCallback, this, std::placeholders::_1));

  	o_tfpose = std::make_unique<tf2_ros::TransformBroadcaster>(*this);


		RCLCPP_INFO(this->get_logger(), "ConfigNMCL::Ready!");    
 
	}    

	
 
	void motionCallback(const nav_msgs::msg::Odometry::SharedPtr odom)
	{
		if (o_firstOdom)
		{
			o_firstOdom = false;
			o_prevPose = OdomMsg2Pose2D(odom);
			return;
		}

		Eigen::Vector3f currPose = OdomMsg2Pose2D(odom);
		Eigen::Vector3f delta = currPose - o_prevTriggerPose;
		Eigen::Vector3f u = o_renmcl->Backward(o_prevPose, currPose);  

		std::vector<Eigen::Vector3f> command{u};
		o_mtx->lock();
		o_renmcl->Predict(command, o_odomWeights, o_odomNoise);
		SetStatistics stas = o_renmcl->Stats();
		o_mtx->unlock(); 

		o_prevPose = currPose; 
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


	void omniCallback(const std_msgs::msg::Float32MultiArray::SharedPtr obj3dMsg)
 	{
 		std::vector<int> labels;
		std::vector<std::vector<Eigen::Vector3f>> vertices;
		std::vector<float> confidences;
		
		//RCLCPP_INFO(this->get_logger(), "ConfigNMCL::Callback!"); 
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
				RCLCPP_INFO(this->get_logger(), "ConfigNMCL:Failure!"); 
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
		
		geometry_msgs::msg::PoseWithCovarianceStamped poseStamped = Pred2PoseWithCov(pred, cov);
		poseStamped.header.frame_id = o_mapTopic;
		poseStamped.header.stamp = this->get_clock()->now();  
		o_posePub->publish(poseStamped); 

		geometry_msgs::msg::PoseArray posearray;
		posearray.header.stamp = this->get_clock()->now();   
		posearray.header.frame_id = o_mapTopic;
		posearray.poses = std::vector<geometry_msgs::msg::Pose>(particles.size());

		for (long unsigned int i = 0; i < particles.size(); ++i)
		{
			geometry_msgs::msg::Pose p;
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

		o_particlePub->publish(posearray);

		tf2::Transform tf_orig;
		tf_orig.setOrigin(tf2::Vector3(pred(0), pred(1), 0.0));
		tf2::Quaternion q;
		q.setRPY(0, 0, pred(2));
		tf_orig.setRotation(q);
		tf2::Transform tf_inv = tf_orig.inverse();
		geometry_msgs::msg::TransformStamped t;
		t.header.stamp = this->get_clock()->now();  
		t.header.frame_id = o_baseLinkTF.c_str();
		t.child_frame_id = o_mapTopic.c_str(); 
		t.transform.translation.x = tf_inv.getOrigin().x();
		t.transform.translation.y = tf_inv.getOrigin().y();
		t.transform.translation.z = tf_inv.getOrigin().z();
		t.transform.rotation.x = tf_inv.getRotation().x();
		t.transform.rotation.y = tf_inv.getRotation().y();
		t.transform.rotation.z = tf_inv.getRotation().z();
		t.transform.rotation.w = tf_inv.getRotation().w();
		o_tfpose->sendTransform(t);
	}

private:

	rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr o_posePub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr o_particlePub;
	rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr o_omniSub;
	rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr o_odomSub;

	std::unique_ptr<tf2_ros::TransformBroadcaster> o_tfpose;


	std::vector<float> o_odomWeights = {1.0};
	Eigen::Vector3f o_prevPose = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f o_prevTriggerPose = Eigen::Vector3f(0, 0, 0);

	Eigen::Vector3f o_odomNoise = Eigen::Vector3f(0.02, 0.02, 0.02);
	Eigen::Vector3d o_pred = Eigen::Vector3d(0, 0, 0);
	Eigen::Matrix3d o_cov;

	bool o_firstOdom = true;
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



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ConfigNMCLNode>());
  rclcpp::shutdown();
  return 0;
}