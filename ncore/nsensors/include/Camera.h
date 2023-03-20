/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: Camera.h                                                              #
# ##############################################################################
**/

#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>


class Camera
{
	public:
	//! A constructor for Camera object, that maps from image coordinates to 3D in camera frame 
	    /*!
	      \param k 
	      \param tc
	    */
		Camera(Eigen::Matrix3d k, Eigen::Matrix3d t);

		Camera(std::string jsonPath);

		std::pair<Eigen::Vector3d, Eigen::Vector3d> UV2CameraFrame(Eigen::Vector2d q1, Eigen::Vector2d q2);

		Eigen::Vector3d UV2CameraFrame(Eigen::Vector2f q1);

		std::pair<float, float> ComputeOccAngles(Eigen::Vector2d q1, Eigen::Vector2d q2);

		int ID() const
		{
			return o_id;
		}


		float Yaw() const
		{
			return o_yaw;
		}

		


	private:

		Eigen::Matrix3d o_K;
		Eigen::Matrix3d o_invK;
		Eigen::Matrix3d o_T;
		int o_id = 0;
		float o_yaw = 0.0;
};


#endif