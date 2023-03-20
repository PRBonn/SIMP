/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SetStatistics.h          			                           		   #
# ##############################################################################
**/


#ifndef SETSTATISTICS_H
#define SETSTATISTICS_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include "Particle.h"

class SetStatistics
{
	public:

		SetStatistics(Eigen::Vector3d m = Eigen::Vector3d::Zero(), Eigen::Matrix3d c = Eigen::Matrix3d::Zero(), unsigned int id = 0)
		{
			mean = m;
			cov = c;
			floorId = id;
		}

		Eigen::Vector3d Mean()
		{
			return mean;
		}

		Eigen::Matrix3d Cov()
		{
			return cov;
		}

		unsigned int FloorID()
		{
			return floorId;
		}

		void FloorID(unsigned int id)
		{
			floorId = id;
		}

		static SetStatistics ComputeParticleSetStatistics(const std::vector<Particle>& particles);

	private:

		Eigen::Vector3d mean;
		Eigen::Matrix3d cov;
		unsigned int floorId;
	
};


#endif