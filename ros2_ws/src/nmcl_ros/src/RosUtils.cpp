/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: RosUtils.cpp                                                          #
# ##############################################################################
**/


#include "RosUtils.h"
#include "Utils.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
//#include <cv_bridge/cv_bridge.h>


geometry_msgs::msg::PoseStamped Pose2D2PoseMsg(Eigen::Vector3f pose2d)
{
    geometry_msgs::msg::PoseStamped poseStamped;
    poseStamped.pose.position.x = pose2d(0);
    poseStamped.pose.position.y = pose2d(1);
    poseStamped.pose.position.z = 0;
    tf2::Quaternion q;
    q.setRPY(0.0,0.0, pose2d(2));
    q = q.normalize();

    poseStamped.pose.orientation.x = q[0];
    poseStamped.pose.orientation.y = q[1];
    poseStamped.pose.orientation.z = q[2];
    poseStamped.pose.orientation.w = q[3];

    return poseStamped;
}


Eigen::Vector3f OdomMsg2Pose2D(const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
{
    float x = odom->pose.pose.position.x;
    float y = odom->pose.pose.position.y;
    float qz = odom->pose.pose.orientation.z;
    float qw = odom->pose.pose.orientation.w;

    Eigen::Vector3f pose = Eigen::Vector3f(x, y, GetYaw(qz, qw));

    return pose;
}

Eigen::Vector3f PoseMsg2Pose2D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& poseMsg)
{
    float x = poseMsg->pose.position.x;
    float y = poseMsg->pose.position.y;
    float qz = poseMsg->pose.orientation.z;
    float qw = poseMsg->pose.orientation.w;

    Eigen::Vector3f pose = Eigen::Vector3f(x, y, GetYaw(qz, qw));

    return pose;
}

std::vector<float> OdomMsg2Pose3D(const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
{

    std::vector<float> pose;
    pose.push_back(odom->pose.pose.position.x);
    pose.push_back(odom->pose.pose.position.y);
    pose.push_back(odom->pose.pose.position.y);
    pose.push_back(odom->pose.pose.orientation.x);
    pose.push_back(odom->pose.pose.orientation.y);
    pose.push_back(odom->pose.pose.orientation.z);
    pose.push_back(odom->pose.pose.orientation.w);
    
    return pose;

}

std::vector<float> PoseMsg2Pose3D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& odom)
{

    std::vector<float> pose;
    pose.push_back(odom->pose.position.x);
    pose.push_back(odom->pose.position.y);
    pose.push_back(odom->pose.position.y);
    pose.push_back(odom->pose.orientation.x);
    pose.push_back(odom->pose.orientation.y);
    pose.push_back(odom->pose.orientation.z);
    pose.push_back(odom->pose.orientation.w);
    
    return pose;
}

pcl::PointCloud<pcl::PointXYZRGB> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d, Eigen::Vector3f rgb)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = points3d.size();
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        cloud.points[i].x = points3d[i](0);
        cloud.points[i].y = points3d[i](1);
        cloud.points[i].z = points3d[i](2);
        cloud.points[i].r = rgb(0);
        cloud.points[i].g = rgb(1);
        cloud.points[i].b = rgb(2);
    }
 
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = points3d.size();
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        cloud.points[i].x = points3d[i](0);
        cloud.points[i].y = points3d[i](1);
        cloud.points[i].z = points3d[i](2);
    }
 
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> ManyVec2PointCloud(const std::vector<std::vector<Eigen::Vector3f>>& points3d, std::vector<Eigen::Vector3f> rgb)
{

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    int pclSize = 0;
    for (size_t i = 0; i < points3d.size(); ++i)
    {
        pclSize += points3d[i].size();
    }
    //std::cout << "pclSize :" << pclSize << std::endl;
    cloud.width = pclSize;
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);

    int cnt = 0;
    for (size_t j = 0; j < points3d.size(); ++j)
    {
        std::vector<Eigen::Vector3f> p3d = points3d[j];
        Eigen::Vector3f clr = rgb[j];

        for (size_t i = 0; i < p3d.size(); ++i)
        {
            cloud.points[cnt + i].x = p3d[i](0);
            cloud.points[cnt + i].y = p3d[i](1);
            cloud.points[cnt + i].z = p3d[i](2);
            cloud.points[cnt + i].r = clr(0);
            cloud.points[cnt + i].g = clr(1);
            cloud.points[cnt + i].b = clr(2);
        }
        cnt += p3d.size();
    }
 
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> Vec2RGBPointCloud(const std::vector<Eigen::Vector3f>& points3d, std::vector<Eigen::Vector3f> rgb)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = points3d.size();
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        cloud.points[i].x = points3d[i](0);
        cloud.points[i].y = points3d[i](1);
        cloud.points[i].z = points3d[i](2);
        cloud.points[i].r = rgb[i](0);
        cloud.points[i].g = rgb[i](1);
        cloud.points[i].b = rgb[i](2);
    }
 
    return cloud;
}

geometry_msgs::msg::PoseWithCovarianceStamped Pred2PoseWithCov(Eigen::Vector3d pred, Eigen::Matrix3d cov)
{

    geometry_msgs::msg::PoseWithCovarianceStamped poseStamped;
 
    poseStamped.pose.pose.position.x = pred(0);
    poseStamped.pose.pose.position.y = pred(1);
    poseStamped.pose.pose.position.z = 0.1;
    tf2::Quaternion q;
    q.setRPY(0.0,0.0, pred(2));
    q = q.normalize();

    poseStamped.pose.pose.orientation.x = q[0];
    poseStamped.pose.pose.orientation.y = q[1];
    poseStamped.pose.pose.orientation.z = q[2];
    poseStamped.pose.pose.orientation.w = q[3];

    // float covariance[36] = {0};

    // // x-x
    // covariance[0] = cov(0, 0);
    // // x-y
    // covariance[1] = cov(0, 1);
    // // y-x
    // covariance[6] = cov(1, 0);
    // // y-y
    // covariance[7] = cov(1, 1);
    // // yaw-yaw
    // covariance[35] = cov(2, 2);

    poseStamped.pose.covariance[0] = cov(0, 0);
    poseStamped.pose.covariance[1] = cov(0, 1);
    poseStamped.pose.covariance[6] = cov(1, 0);
    poseStamped.pose.covariance[7] = cov(1, 1);
    poseStamped.pose.covariance[35] = cov(2, 2);


    return poseStamped;
}


// sensor_msgs::msg::Image CVMat2ImgMsg(const cv::Mat& img, std_msgs::Header header, const std::string& type)
// {
//     cv_bridge::CvImage img_bridge;
//     sensor_msgs::msg::Image imgMsg;
//     img_bridge = cv_bridge::CvImage(header, type, img);
//     img_bridge.toImageMsg(imgMsg); 

//     return imgMsg;
// }







// geometry_msgs::msg::PoseStamped Pose2D2PoseMsg(Eigen::Vector3f pose2d)
// {
//     geometry_msgs::msg::PoseStamped poseStamped;
//     poseStamped.pose.position.x = pose2d(0);
//     poseStamped.pose.position.y = pose2d(1);
//     poseStamped.pose.position.z = 0;
//     tf2::Quaternion q;
//     q.setRPY(0.0,0.0, pose2d(2));
//     q = q.normalize();

//     poseStamped.pose.orientation.x = q[0];
//     poseStamped.pose.orientation.y = q[1];
//     poseStamped.pose.orientation.z = q[2];
//     poseStamped.pose.orientation.w = q[3];

//     return poseStamped;
// }


// Eigen::Vector3f OdomMsg2Pose2D(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
// {
// 	float x = odom->pose.pose.position.x;
// 	float y = odom->pose.pose.position.y;
// 	float qz = odom->pose.pose.orientation.z;
// 	float qw = odom->pose.pose.orientation.w;

// 	Eigen::Vector3f pose = Eigen::Vector3f(x, y, GetYaw(qz, qw));

// 	return pose;
// }

// Eigen::Vector3f PoseMsg2Pose2D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& poseMsg)
// {
//     float x = poseMsg->pose.position.x;
//     float y = poseMsg->pose.position.y;
//     float qz = poseMsg->pose.orientation.z;
//     float qw = poseMsg->pose.orientation.w;

//     Eigen::Vector3f pose = Eigen::Vector3f(x, y, GetYaw(qz, qw));

//     return pose;
// }

// std::vector<float> OdomMsg2Pose3D(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
// {

//     std::vector<float> pose;
//     pose.push_back(odom->pose.pose.position.x);
//     pose.push_back(odom->pose.pose.position.y);
//     pose.push_back(odom->pose.pose.position.y);
//     pose.push_back(odom->pose.pose.orientation.x);
//     pose.push_back(odom->pose.pose.orientation.y);
//     pose.push_back(odom->pose.pose.orientation.z);
//     pose.push_back(odom->pose.pose.orientation.w);
    
//     return pose;

// }

// std::vector<float> PoseMsg2Pose3D(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& odom)
// {

//     std::vector<float> pose;
//     pose.push_back(odom->pose.position.x);
//     pose.push_back(odom->pose.position.y);
//     pose.push_back(odom->pose.position.y);
//     pose.push_back(odom->pose.orientation.x);
//     pose.push_back(odom->pose.orientation.y);
//     pose.push_back(odom->pose.orientation.z);
//     pose.push_back(odom->pose.orientation.w);
    
//     return pose;
// }
// /*
// pcl::PointCloud<pcl::PointXYZRGB> Vec2PointCloud(const std::vector<Eigen::Vector3f>& points3d, Eigen::Vector3f rgb)
// {
// 	pcl::PointCloud<pcl::PointXYZRGB> cloud;
// 	cloud.width = points3d.size();
//     cloud.height = 1;
//     cloud.points.resize(cloud.width * cloud.height);
//     for (size_t i = 0; i < cloud.points.size(); ++i)
//     {
//         cloud.points[i].x = points3d[i](0);
//         cloud.points[i].y = points3d[i](1);
//         cloud.points[i].z = points3d[i](2);
//         cloud.points[i].r = rgb(0);
//         cloud.points[i].g = rgb(1);
//         cloud.points[i].b = rgb(2);
//     }
 
//     return cloud;
// }*/

// geometry_msgs::msg::PoseWithCovarianceStamped Pred2PoseWithCov(Eigen::Vector3d pred, Eigen::Matrix3d cov)
// {

//     geometry_msgs::msg::PoseWithCovarianceStamped poseStamped;
 
//     poseStamped.pose.pose.position.x = pred(0);
//     poseStamped.pose.pose.position.y = pred(1);
//     poseStamped.pose.pose.position.z = 0;
//     tf2::Quaternion q;
//     q.setRPY(0.0,0.0, pred(2));
//     q = q.normalize();

//     poseStamped.pose.pose.orientation.x = q[0];
//     poseStamped.pose.pose.orientation.y = q[1];
//     poseStamped.pose.pose.orientation.z = q[2];
//     poseStamped.pose.pose.orientation.w = q[3];

//     // float covariance[36] = {0};

//     // // x-x
//     // covariance[0] = cov(0, 0);
//     // // x-y
//     // covariance[1] = cov(0, 1);
//     // // y-x
//     // covariance[6] = cov(1, 0);
//     // // y-y
//     // covariance[7] = cov(1, 1);
//     // // yaw-yaw
//     // covariance[35] = cov(2, 2);

//     poseStamped.pose.covariance[0] = cov(0, 0);
//     poseStamped.pose.covariance[1] = cov(0, 1);
//     poseStamped.pose.covariance[6] = cov(1, 0);
//     poseStamped.pose.covariance[7] = cov(1, 1);
//     poseStamped.pose.covariance[35] = cov(2, 2);


//     return poseStamped;
// }