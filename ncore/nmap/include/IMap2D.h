/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: IMap2D.h                                                              #
# ##############################################################################
**/

#ifndef IMAP2D_H
#define IMAP2D_H

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>

class IMap2D
{
	public:


		//! Converts (x, y) from the map frame to the pixels coordinates
		/*!
			\param (x, y) position in map frame
		   \return (u, v) pixel coordinates for the gridmap
		*/
		virtual Eigen::Vector2f World2Map(Eigen::Vector2f xy) const = 0;
		

		//! Converts (u, v) pixel coordinates to map frame (x, y)
		/*!
			\param (u, v) pixel coordinates for the gridmap
		   \return (x, y) position in map frame
		*/
		virtual Eigen::Vector2f Map2World(Eigen::Vector2f uv) const = 0 ;


		virtual bool IsValid(Eigen::Vector3f pose) const = 0 ;

		virtual bool IsValid2D(Eigen::Vector2f mp) const = 0 ;


		virtual ~IMap2D() {};

		
		virtual const cv::Mat& Map() const = 0;

	
};

#endif // IMap2D