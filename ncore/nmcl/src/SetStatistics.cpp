/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SetStatistics.cpp                		                               #
# ##############################################################################
**/

#include "SetStatistics.h"
#include <iostream>

SetStatistics SetStatistics::ComputeParticleSetStatistics(const std::vector<Particle>& particles)
{

	Eigen::Vector4d m = Eigen::Vector4d(0, 0, 0, 0);
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero(3, 3);
	int n = particles.size();
	double tot_w = 0.0;

	for(int i = 0; i < n; ++i)
	{
		Eigen::Vector3f p = particles[i].pose;
		double w = particles[i].weight;

		m(0) += p(0) * w; // mean x
		m(1) += p(1) * w; // mean y
		m(2) += cos(p(2)) * w; // theta 
		m(3) += sin(p(2)) * w; // theta

		tot_w += w;

		// linear components cov
		for(int j = 0; j < 2; ++j)
		{
			for(int k = 0; k < 2; ++k)
			{
				 cov(j,k) += w * p(j) * p(k);
			}
		}
	}

	Eigen::Vector3d mean;
	mean(0) = m(0) / tot_w; 
	mean(1) = m(1) / tot_w; 
	mean(2) = atan2(m(3), m(2)); 


	// normalize linear components cov
	for(int j = 0; j < 2; ++j)
	{
		for(int k = 0; k < 2; ++k)
		{
			 cov(j, k) = cov(j, k) /tot_w - mean(j) * mean(k);
		}
	}

	// angular covariance
	double R = sqrt(m(2) * m(2) + m(3) * m(3));

	// https://github.com/ros-planning/navigation/blob/2b807bd312fac1b476851800c84cb962559cbc53/amcl/src/amcl/pf/pf.c#L690
	//cov(2, 2) = -2 * log(R);

	// https://www.ebi.ac.uk/thornton-srv/software/PROCHECK/nmr_manual/man_cv.html
	cov(2, 2) = 1 - R / tot_w;


	SetStatistics stats = SetStatistics(mean, cov);

	return stats;

}