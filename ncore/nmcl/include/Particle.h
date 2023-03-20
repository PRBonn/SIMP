/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: Particle.h           				                           		   #
# ##############################################################################
**/

#ifndef PARTICLE_H
#define PARTICLE_H


#include <vector>
#include <eigen3/Eigen/Dense>


class Particle
{
public:

	Particle(Eigen::Vector3f p = Eigen::Vector3f(0, 0, 0), double w = 0, unsigned int id = 0);
	
	Eigen::Vector3f pose;
	unsigned int floorID;
	double weight;

};


#endif