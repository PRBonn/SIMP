/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: MSensorsUnitTests.cpp                                                 #
# ##############################################################################
**/

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <string>


#include "Utils.h"
#include "Lidar2D.h"
#include "OptiTrack.h"
#include "Camera.h"

std::string dataPath = PROJECT_TEST_DATA_DIR + std::string("/8/");
std::string configPath = PROJECT_TEST_DATA_DIR + std::string("/config/");



TEST(TestUtils, test1) {
    
    float angle = 2 * M_PI;
	//should be 0 or near
    float angle_wrap = Wrap2Pi(angle);
	ASSERT_NEAR(0.0, angle_wrap, 0.000001);
}

TEST(TestUtils, test2) {
    
	// identity martrix
	Eigen::Matrix3f identity = Eigen::Matrix3f::Identity(3, 3);
	Eigen::Matrix3f v2t = Vec2Trans(Eigen::Vector3f(0, 0, 0));
	ASSERT_EQ(identity, v2t);

}

TEST(TestUtils, test3) {
    
	// from python verified code, should be -1.5707963267948966
	float yaw = GetYaw(0.7, -0.7);
	ASSERT_NEAR(yaw, -1.5707963267948966, 0.00001);

}

TEST(TestUtils, test4) {
    
	std::vector<float> ranges{1.0, 2.0, 3.0};
	std::vector<float> angles{1.0, 0.0, -0.5};
	std::vector<Eigen::Vector3f> hp = Ranges2Points(ranges, angles);

	// from python verified code
	/*
		hp = [[ 0.54030231  2.          2.63274769]
		 	  [ 0.84147098  0.         -1.43827662]
		 	  [ 1.          1.          1.        ]]
	*/

	std::vector<Eigen::Vector3f> hp_gt{Eigen::Vector3f(0.54030231, 0.84147098, 1.0),
										Eigen::Vector3f(2.0, 0.0, 1.0), 
										Eigen::Vector3f(2.63274769, -1.43827662, 1.0)};

	ASSERT_NEAR(hp[0](0), hp_gt[0](0), 0.001);
	ASSERT_NEAR(hp[0](1), hp_gt[0](1), 0.001);
	ASSERT_NEAR(hp[0](2), hp_gt[0](2), 0.001);
	ASSERT_NEAR(hp[1](0), hp_gt[1](0), 0.001);
	ASSERT_NEAR(hp[1](1), hp_gt[1](1), 0.001);
	ASSERT_NEAR(hp[1](2), hp_gt[1](2), 0.001);
	ASSERT_NEAR(hp[2](0), hp_gt[2](0), 0.001);
	ASSERT_NEAR(hp[2](1), hp_gt[2](1), 0.001);
	ASSERT_NEAR(hp[2](2), hp_gt[2](2), 0.001);

}

TEST(TestUtils, test5) {
    
	std::vector<float> ranges{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	std::vector<float> downsampled = Downsample(ranges, 3);
	std::vector<float> downsampled_gt = {1.0, 4.0, 7.0, 10.0, 13.0};
	ASSERT_EQ(downsampled, downsampled_gt);
}


TEST(TestCamera, test1) {
    
	Eigen::Matrix3d tc;
	Eigen::Matrix3d invK;
	Eigen::Matrix3d k;
	k << 614.6570434570312, 0.0, 326.6780090332031, 0.0, 614.308349609375, 242.25645446777344, 0.0, 0.0, 1.0;
	tc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
	invK << 0.00162692, 0, -0.53148, 0, 0.00162785, -0.394356, 0, 0, 1;
	Camera cam = Camera(k, tc);

	Eigen::Vector2d q1(0, 100);
	Eigen::Vector2d q2(400, 100);
	std::pair<Eigen::Vector3d, Eigen::Vector3d> pc = cam.UV2CameraFrame(q1, q2);

	Eigen::Vector3d p1c = pc.first;
	Eigen::Vector3d p2c = pc.second;


	// gt results from python [1.       0.53148  0.231571] [ 1.       -0.119288  0.231571]
	Eigen::Vector3d p1c_gt(1., 0.53148, 0.231571);
	Eigen::Vector3d p2c_gt(1., -0.119288,  0.231571);

	ASSERT_NEAR(p1c(0), p1c_gt(0), 0.00001);
	ASSERT_NEAR(p1c(1), p1c_gt(1), 0.00001);
	ASSERT_NEAR(p1c(2), p1c_gt(2), 0.00001);
	ASSERT_NEAR(p2c(0), p2c_gt(0), 0.00001);
	ASSERT_NEAR(p2c(1), p2c_gt(1), 0.00001);
	ASSERT_NEAR(p2c(2), p2c_gt(2), 0.00001);
}

TEST(TestCamera, test2) {
    
	Eigen::Matrix3d tc;
	Eigen::Matrix3d invK;
	Eigen::Matrix3d k;
	k << 614.6570434570312, 0.0, 326.6780090332031, 0.0, 614.308349609375, 242.25645446777344, 0.0, 0.0, 1.0;
	tc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
	invK << 0.00162692, 0, -0.53148, 0, 0.00162785, -0.394356, 0, 0, 1;
	Camera cam = Camera(configPath + "cam0.config");

	Eigen::Vector2d q1(0, 100);
	Eigen::Vector2d q2(400, 100);
	std::pair<Eigen::Vector3d, Eigen::Vector3d> pc = cam.UV2CameraFrame(q1, q2);

	Eigen::Vector3d p1c = pc.first;
	Eigen::Vector3d p2c = pc.second;


	// gt results from python [1.       0.53148  0.231571] [ 1.       -0.119288  0.231571]
	Eigen::Vector3d p1c_gt(1., 0.53148, 0.231571);
	Eigen::Vector3d p2c_gt(1., -0.119288,  0.231571);

	ASSERT_NEAR(p1c(0), p1c_gt(0), 0.00001);
	ASSERT_NEAR(p1c(1), p1c_gt(1), 0.00001);
	ASSERT_NEAR(p1c(2), p1c_gt(2), 0.00001);
	ASSERT_NEAR(p2c(0), p2c_gt(0), 0.00001);
	ASSERT_NEAR(p2c(1), p2c_gt(1), 0.00001);
	ASSERT_NEAR(p2c(2), p2c_gt(2), 0.00001);
}

TEST(TestOptiTrack, test1)
{
	Eigen::Vector3f o = Eigen::Vector3f(1.3097, 0.5226, -3.138);
	OptiTrack op = OptiTrack(o);
	Eigen::Vector3f p_trans = op.OptiTrack2World(o);
	
	ASSERT_NEAR(p_trans(0), 0.0, 0.00001);
	ASSERT_NEAR(p_trans(1), 0.0, 0.00001);
	ASSERT_NEAR(p_trans(2), 0.0, 0.00001);

}

TEST(TestOptiTrack, test2)
{
	Eigen::Vector3f o = Eigen::Vector3f(1.3097140789031982, 0.5226072072982788, -3.138);
	OptiTrack op = OptiTrack(dataPath);
	Eigen::Vector3f p_trans = op.OptiTrack2World(o);
	
	ASSERT_NEAR(p_trans(0), 0.0, 0.00001);
	ASSERT_NEAR(p_trans(1), 0.0, 0.00001);
	ASSERT_NEAR(p_trans(2), 0.0, 0.00001);

}


TEST(TestLidar2D, test1)
{
	float maxAngle = 2.268928;
	float minAngle = -2.268928;
	int nBeams = 1041;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), nBeams, maxAngle, minAngle);
	std::vector<float> heading = l2d_f.Heading();

	float reso = (maxAngle - minAngle) / (nBeams - 1);

	ASSERT_EQ(heading.size(), nBeams);
	ASSERT_EQ(heading[0], minAngle);
	ASSERT_EQ(heading[nBeams - 1], maxAngle);
	ASSERT_EQ(heading[1], minAngle + reso);
}


TEST(TestLidar2D, test2)
{
	float maxAngle = 2.268928;
	float minAngle = -2.268928;
	int nBeams = 1041;
	Lidar2D l2d_f = Lidar2D("front_laser", configPath);
	std::vector<float> heading = l2d_f.Heading();

	float reso = (maxAngle - minAngle) / (nBeams - 1);

	ASSERT_EQ(heading.size(), nBeams);
	ASSERT_EQ(heading[0], minAngle);
	ASSERT_EQ(heading[nBeams - 1], maxAngle);
	ASSERT_EQ(heading[1], minAngle + reso);
}

TEST(TestLidar2D, test3)
{
	float maxAngle = 2.268928;
	float minAngle = -2.268928;
	int nBeams = 1041;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), nBeams, maxAngle, minAngle);

	std::vector<Eigen::Vector3f> hp{Eigen::Vector3f(0.54030231, 0.84147098, 1.0),
										Eigen::Vector3f(2.0, 0.0, 1.0), 
										Eigen::Vector3f(2.63274769, -1.43827662, 1.0)};

	std::vector<Eigen::Vector3f> tp = l2d_f.Center(hp);
	// from python verified code
	/* 
		tp = [[0.03743063 1.66477654 3.12898496]
 			  [1.13214598 1.56865036 0.99847235]
 			  [1.         1.         1.        ]]
	*/
	std::vector<Eigen::Vector3f> tp_gt{Eigen::Vector3f(0.03743063, 1.13214598, 1.0), 
										Eigen::Vector3f(1.66477654, 1.56865036, 1.0), 
										Eigen::Vector3f(3.12898496, 0.99847235, 1.0)};

	ASSERT_NEAR(tp[0](0), tp_gt[0](0), 0.001);
	ASSERT_NEAR(tp[0](1), tp_gt[0](1), 0.001);
	ASSERT_NEAR(tp[0](2), tp_gt[0](2), 0.001);
	ASSERT_NEAR(tp[1](0), tp_gt[1](0), 0.001);
	ASSERT_NEAR(tp[1](1), tp_gt[1](1), 0.001);
	ASSERT_NEAR(tp[1](2), tp_gt[1](2), 0.001);
	ASSERT_NEAR(tp[2](0), tp_gt[2](0), 0.001);
	ASSERT_NEAR(tp[2](1), tp_gt[2](1), 0.001);
	ASSERT_NEAR(tp[2](2), tp_gt[2](2), 0.001);

}

TEST(TestLidar2D, test4)
{
	float maxAngle = 2.268928;
	float minAngle = -2.268928;
	int nBeams = 1041;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), nBeams, maxAngle, minAngle);
	Lidar2D l2d_r = Lidar2D("rear_laser", Eigen::Vector3f(-0.25, -0.155, -2.356), nBeams, maxAngle, minAngle);

	
	std::vector<float> ranges(nBeams, 1.0);

	std::vector<Eigen::Vector3f> points_3d = MergeScans(ranges, l2d_f, ranges, l2d_r);
	ASSERT_EQ(points_3d.size(), nBeams * 2 / 10);

	// from python verified code
	Eigen::Vector3f p3d0_gt = Eigen::Vector3f(0.33675906, -0.84122932,  1. );


	ASSERT_NEAR(points_3d[0](0), p3d0_gt(0), 0.001);
	ASSERT_NEAR(points_3d[0](1), p3d0_gt(1), 0.001);
	ASSERT_NEAR(points_3d[0](2), p3d0_gt(2), 0.001);

}

TEST(TestLidar2D, test5)
{
	float maxAngle = 2.268928;
	float minAngle = -2.268928;
	int nBeams = 1041;
	Lidar2D l2d_f = Lidar2D(configPath + "front_laser.config");
	std::vector<float> heading = l2d_f.Heading();

	float reso = (maxAngle - minAngle) / (nBeams - 1);

	ASSERT_EQ(heading.size(), nBeams);
	ASSERT_EQ(heading[0], minAngle);
	ASSERT_EQ(heading[nBeams - 1], maxAngle);
	ASSERT_EQ(heading[1], minAngle + reso);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}