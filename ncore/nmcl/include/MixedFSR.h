/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: MixedFSR.h           					                               #
# ##############################################################################
**/


#ifndef MIXEDFSR_H
#define MIXEDFSR_H

#include <eigen3/Eigen/Dense>
#include <vector>

class MixedFSR 
{
	public:

		Eigen::Vector3f SampleMotion(const Eigen::Vector3f& p1, const std::vector<Eigen::Vector3f>& command, const std::vector<float>& weights, const Eigen::Vector3f& noise);

		Eigen::Vector3f Forward(Eigen::Vector3f p1, Eigen::Vector3f u);

	    Eigen::Vector3f Backward(Eigen::Vector3f p1, Eigen::Vector3f p2);


};

#endif