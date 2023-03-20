/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: Lidar2D.h                                                             #
# ##############################################################################
**/

#ifndef LIDAR2D_H
#define LIDAR2D_H

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>


class Lidar2D
{
	public:

		Lidar2D(std::string name_, Eigen::Vector3f origin, int numBeams, float maxAngle, float minAngle);

		Lidar2D(std::string name_, std::string yamlFolder);

		Lidar2D(std::string jsonPath);

		//const std::vector<float>& Heading() const
		std::vector<float> Heading() 
		{
			return o_heading;
		}

		std::string Name()
		{
			return o_name;
		}

		std::vector<Eigen::Vector3f> Center(std::vector<Eigen::Vector3f>& homoPoints);



	private:

		std::vector<float> o_heading;
		std::string o_name;
		Eigen::Matrix3f o_trans;

};



std::vector<Eigen::Vector3f> MergeScans(const std::vector<float>& f_ranges, Lidar2D laser_front, 
	const std::vector<float>& r_ranges, Lidar2D laser_rear, int dsFactor = 10, float maxRange = 15);

std::vector<Eigen::Vector3f> MergeScansSimple(const std::vector<float>& f_ranges, Lidar2D laser_front, 
	const std::vector<float>& r_ranges, Lidar2D laser_rear);

#endif