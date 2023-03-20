/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: Camera.cpp                                                            #
# ##############################################################################
**/


#include "Camera.h"
#include "Utils.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>

Camera::Camera(Eigen::Matrix3d k, Eigen::Matrix3d t)
{
	o_K = k;
	o_invK = o_K.inverse();
	o_T = t;
}

Camera::Camera(std::string jsonPath)
{
	using json = nlohmann::json;

	std::ifstream file(jsonPath);
	json config;
	file >> config;
	o_id = config["id"];
	o_yaw = config["yaw"];
	std::vector<double> k = config["k"];
	std::vector<double> t = config["t"];

	o_K = Eigen::Matrix<double,3,3,Eigen::RowMajor>(k.data());
	o_invK = o_K.inverse();
	o_T = Eigen::Matrix<double,3,3,Eigen::RowMajor>(t.data());
}



std::pair<Eigen::Vector3d, Eigen::Vector3d> Camera::UV2CameraFrame(Eigen::Vector2d q1, Eigen::Vector2d q2)
{
	Eigen::Vector3d p1(q1(0), q1(1), 1);
	Eigen::Vector3d p2(q2(0), q2(1), 1);


    // multiply by inverse calibration matrix
    Eigen::Vector3d p1k_ = o_invK * p1;
    Eigen::Vector3d p2k_ = o_invK * p2;

    // divide by z component to homogenize it
    Eigen::Vector3d p1k = p1k_ / p1k_(2);
    Eigen::Vector3d p2k = p2k_ / p2k_(2);

    // go from image frame to camera frame
    Eigen::Vector3d p1c = o_T * p1k;
    Eigen::Vector3d p2c = o_T * p2k;

    std::pair<Eigen::Vector3d, Eigen::Vector3d> pc{p1c, p2c};

    return pc;
}

Eigen::Vector3d Camera::UV2CameraFrame(Eigen::Vector2f q1)
{
	Eigen::Vector3d p1(q1(0), q1(1), 1);

    // multiply by inverse calibration matrix
    Eigen::Vector3d p1k_ = o_invK * p1;

    // divide by z component to homogenize it
    Eigen::Vector3d p1k = p1k_ / p1k_(2);

    // go from image frame to camera frame
    Eigen::Vector3d p1c = o_T * p1k;

    return p1c;
}


std::pair<float, float> Camera::ComputeOccAngles(Eigen::Vector2d q1, Eigen::Vector2d q2)
{
	std::pair<Eigen::Vector3d, Eigen::Vector3d> pc = UV2CameraFrame(q1, q2);
	Eigen::Vector3d p1c = pc.first;
	Eigen::Vector3d p2c = pc.second;
	float t1 = Wrap2Pi(atan2(p1c(1), p1c(0)) + o_yaw);
	float t2 = Wrap2Pi(atan2(p2c(1), p2c(0)) + o_yaw);

	return std::pair <float, float>(t1, t2);
}