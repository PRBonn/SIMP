/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: RosUtils.h                                                        #
# ##############################################################################
**/


#ifndef ROSUTILS_H
#define ROSUTILS_H

#include <eigen3/Eigen/Dense>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>

//#include <sensor_msgs/msg/point_cloud2.hpp>
//#include <pcl_ros/point_cloud.h>
#include <vector>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>


// Eigen::Vector3f OdomMsg2Pose2D(const nav_msgs::msg::Odometry::ConstSharedPtr odom);

// Eigen::Vector3f PoseMsg2Pose2D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& poseMsg);

// std::vector<float> OdomMsg2Pose3D(const nav_msgs::msg::Odometry::ConstSharedPtr odom);

// std::vector<float> PoseMsg2Pose3D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& odom);

// geometry_msgs::msg::PoseStamped Pose2D2PoseMsg(Eigen::Vector3f pose2d);


// //pcl::PointCloud<pcl::PointXYZRGB> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d, Eigen::Vector3f rgb);


// geometry_msgs::msg::PoseWithCovarianceStamped Pred2PoseWithCov(Eigen::Vector3d pred, Eigen::Matrix3d cov);


Eigen::Vector3f OdomMsg2Pose2D(const nav_msgs::msg::Odometry::ConstSharedPtr& odom);

Eigen::Vector3f PoseMsg2Pose2D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& poseMsg);

std::vector<float> OdomMsg2Pose3D(const nav_msgs::msg::Odometry::ConstSharedPtr& odom);

std::vector<float> PoseMsg2Pose3D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& odom);

geometry_msgs::msg::PoseStamped Pose2D2PoseMsg(Eigen::Vector3f pose2d);


pcl::PointCloud<pcl::PointXYZRGB> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d, Eigen::Vector3f rgb);

pcl::PointCloud<pcl::PointXYZ> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d);

pcl::PointCloud<pcl::PointXYZRGB> Vec2RGBPointCloud(const std::vector<Eigen::Vector3f>& points3d, std::vector<Eigen::Vector3f> rgb);


pcl::PointCloud<pcl::PointXYZRGB> ManyVec2PointCloud(const std::vector<std::vector<Eigen::Vector3f>>& points3d, std::vector<Eigen::Vector3f> rgb);


geometry_msgs::msg::PoseWithCovarianceStamped Pred2PoseWithCov(Eigen::Vector3d pred, Eigen::Matrix3d cov);

//sensor_msgs::msg::Image CVMat2ImgMsg(const cv::Mat& img, std_msgs::Header header, const std::string& type);

#endif