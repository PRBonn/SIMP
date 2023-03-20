/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: LowVarianceResampling.h           		                               #
# ##############################################################################
**/


#ifndef RESAMPLING_H
#define RESAMPLING_H


#include <vector>
#include <eigen3/Eigen/Dense>
#include "Particle.h"

class Resampling
{
	public:	

		void Resample(std::vector<Particle>& particles);

		void SetTH(float th)
		{
			o_th = th;
		}


	private:

		float o_th = 0.5;

};

#endif