/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: OptiTrack.h                                                           #
# ##############################################################################
**/

#ifndef OPTITRACK_H
#define OPTITRACK_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>

class OptiTrack
{
	public:

		//! A constructor
	    /*!
	      \param origin is the (x, y, theta) location of the robot in the OptiTrack frame when the robot begins mapping
	      		So it is (0, 0, 0) in the map frame. 
	    */
		OptiTrack(Eigen::Vector3f origin);

		//! A constructor
	    /*!
	      \param yamlPath is a string of the path of the folder of a yaml file containing the origin location
	    */
		OptiTrack(std::string yamlFolder);

		//! Converts 2D pose (x, y, theta) in the OptiTrack frame to 2D pose in the map frame. 
		/*!
		  \param p is 2D pose from the OptiTrack motion capture
	      \return 2D pose in the map frame
		*/
		Eigen::Vector3f OptiTrack2World(Eigen::Vector3f p);


	private: 
		Eigen::Matrix3f o_invTrans;
		Eigen::Vector3f o_origin;
};

#endif