/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: OptiTrack.cpp                                                         #
# ##############################################################################
**/


#include "OptiTrack.h"
#include "Utils.h"


OptiTrack::OptiTrack(Eigen::Vector3f origin_)
{
	Eigen::Matrix3f trans = Vec2Trans(origin_);
	o_invTrans = trans.inverse();
	o_origin = origin_;
}

OptiTrack::OptiTrack(std::string yamlFolder)
{
	std::vector<std::string> fields =  File2Lines(yamlFolder + "optitrack.yaml");
	// "origin:" - 8
	fields[0].erase(0,8);
	std::vector<float> vec = StringToVec(fields[0]);

	o_origin = Eigen::Vector3f(vec[0], vec[1], vec[2]);
	Eigen::Matrix3f trans = Vec2Trans(o_origin);
	o_invTrans = trans.inverse();
}



Eigen::Vector3f OptiTrack::OptiTrack2World(Eigen::Vector3f p)
{
	Eigen::Vector3f xy1 = Eigen::Vector3f(p(0), p(1), 1);
	Eigen::Vector3f p_trans = o_invTrans * xy1;
	p_trans(2) = Wrap2Pi(p(2) - o_origin(2));

	return p_trans;
}