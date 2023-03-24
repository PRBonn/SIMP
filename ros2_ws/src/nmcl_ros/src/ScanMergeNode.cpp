/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ScanMergeNode.cpp                                                        #
# ##############################################################################
**/

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>


#include "Utils.h"
#include "Lidar2D.h"
#include "RosUtils.h"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
//#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
//#include <tf/transform_broadcaster.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::LaserScan, sensor_msgs::msg::LaserScan> LidarSyncPolicy;

class ScanMergeNode : public rclcpp::Node
{
  public:
    ScanMergeNode()
    : Node("ScanMergeNode")
    {
      std::string configFolder;
      std::string scanFrontTopic;
      std::string scanRearTopic;

      this->declare_parameter("configFolder");
      this->declare_parameter("scanFrontTopic");
      this->declare_parameter("scanRearTopic");
      this->declare_parameter("baseLinkTF");

      this->get_parameter("configFolder", configFolder);
      RCLCPP_INFO(this->get_logger(), "configFolder %s", configFolder.c_str());
      this->get_parameter("scanFrontTopic", scanFrontTopic);
      RCLCPP_INFO(this->get_logger(), "scanFrontTopic %s", scanFrontTopic.c_str());
      this->get_parameter("scanRearTopic", scanRearTopic);
      RCLCPP_INFO(this->get_logger(), "scanRearTopic %s", scanRearTopic.c_str());
      this->get_parameter("baseLinkTF", o_baseLinkTF);
      RCLCPP_INFO(this->get_logger(), "baseLinkTF %s", o_baseLinkTF.c_str());
    
      
      std::string fldrName =  "front_laser";      
      std::string rldrName =  "rear_laser";  


      l2d_f = std::make_shared<Lidar2D>(Lidar2D(fldrName, configFolder));
      l2d_r = std::make_shared<Lidar2D>(Lidar2D(rldrName, configFolder));

      o_scanPub =  this->create_publisher<sensor_msgs::msg::PointCloud2>("scan_merged", 1);
      tf_scan = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
      laserFrontSub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(this, scanFrontTopic);
      laserRearSub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(this, scanRearTopic);


      lidarSync = std::make_shared<message_filters::Synchronizer<LidarSyncPolicy>>(LidarSyncPolicy(10), *laserFrontSub, *laserRearSub);
      lidarSync->registerCallback(std::bind(&ScanMergeNode::callback, this, std::placeholders::_1, std::placeholders::_2)); 


       RCLCPP_INFO(this->get_logger(), "ScanMergeNode running!");
      
      
    }

    void callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserFront, const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserRear)
    {
      std::vector<float> scanFront = laserFront->ranges;
      std::vector<float> scanRear = laserRear->ranges;

      std::vector<Eigen::Vector3f> points_3d = MergeScansSimple(scanFront, *l2d_f, scanRear, *l2d_r);


      // geometry_msgs::msg::TransformStamped t;
      // t.header.stamp = rclcpp::Time();  
      // t.header.frame_id =  o_baseLinkTF.c_str();
      // t.child_frame_id = "scan_merged"; 
      // t.transform.translation.x = 0.0;
      // t.transform.translation.y = 0.0;
      // t.transform.translation.z = 0.0;
      // tf2::Quaternion q;
      // q.setRPY(0, 0, 0);
      // t.transform.rotation.x = q.x();
      // t.transform.rotation.y = q.y();
      // t.transform.rotation.z = q.z();
      // t.transform.rotation.w = q.w();
      // tf_scan->sendTransform(t); 

      pcl::PointCloud<pcl::PointXYZ> pcl = Vec2PointCloud(points_3d);
      sensor_msgs::msg::PointCloud2 pcl_msg;
      pcl::toROSMsg(pcl, pcl_msg);
      pcl_msg.header.stamp = rclcpp::Time();
      pcl_msg.header.frame_id = "merged_scan";
      o_scanPub->publish(pcl_msg);


      tf2::Transform tf_orig;
      tf_orig.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, 0.0);
      tf_orig.setRotation(q);
      tf2::Transform tf_inv = tf_orig.inverse();
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = this->get_clock()->now();  
      t.header.frame_id = o_baseLinkTF.c_str();
      t.child_frame_id = "merged_scan"; 
      t.transform.translation.x = tf_inv.getOrigin().x();
      t.transform.translation.y = tf_inv.getOrigin().y();
      t.transform.translation.z = tf_inv.getOrigin().z();
      t.transform.rotation.x = tf_inv.getRotation().x();
      t.transform.rotation.y = tf_inv.getRotation().y();
      t.transform.rotation.z = tf_inv.getRotation().z();
      t.transform.rotation.w = tf_inv.getRotation().w();
      tf_scan->sendTransform(t);

    }

  private:

    std::shared_ptr<Lidar2D> l2d_f;
    std::shared_ptr<Lidar2D> l2d_r;
    std::string o_baseLinkTF;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr o_scanPub;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> laserFrontSub;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> laserRearSub;
    std::shared_ptr<message_filters::Synchronizer<LidarSyncPolicy>> lidarSync;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_scan;


   

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanMergeNode>());
  rclcpp::shutdown();
  return 0;
}