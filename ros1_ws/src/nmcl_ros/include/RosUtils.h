/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: RosUtils.h                                                            #
# ##############################################################################
**/

#ifndef ROSUTILS_H
#define ROSUTILS_H

#include <eigen3/Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <vector>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include "opencv2/opencv.hpp"
#include <cv_bridge/cv_bridge.h>


Eigen::Vector3f OdomMsg2Pose2D(const nav_msgs::OdometryConstPtr& odom);

Eigen::Vector3f PoseMsg2Pose2D(const geometry_msgs::PoseStampedConstPtr& poseMsg);

std::vector<float> OdomMsg2Pose3D(const nav_msgs::OdometryConstPtr& odom);

std::vector<float> PoseMsg2Pose3D(const geometry_msgs::PoseStampedConstPtr& odom);

geometry_msgs::PoseStamped Pose2D2PoseMsg(Eigen::Vector3f pose2d);


pcl::PointCloud<pcl::PointXYZRGB> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d, Eigen::Vector3f rgb);

pcl::PointCloud<pcl::PointXYZ> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d);

pcl::PointCloud<pcl::PointXYZRGB> Vec2RGBPointCloud(const std::vector<Eigen::Vector3f>& points3d, std::vector<Eigen::Vector3f> rgb);


pcl::PointCloud<pcl::PointXYZRGB> ManyVec2PointCloud(const std::vector<std::vector<Eigen::Vector3f>>& points3d, std::vector<Eigen::Vector3f> rgb);


geometry_msgs::PoseWithCovarianceStamped Pred2PoseWithCov(Eigen::Vector3d pred, Eigen::Matrix3d cov);

sensor_msgs::Image CVMat2ImgMsg(const cv::Mat& img, std_msgs::Header header, const std::string& type);




#endif
