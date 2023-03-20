/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: MixedFSR.cpp                			                               #
# ##############################################################################
**/

#include "MixedFSR.h"
#include <math.h>
#include <stdlib.h>
#include "Utils.h"
#include <iostream>



Eigen::Vector3f MixedFSR::SampleMotion(const Eigen::Vector3f& p1, const std::vector<Eigen::Vector3f>& command, const std::vector<float>& weights, const Eigen::Vector3f& noise)
{
	Eigen::Vector3f u(0, 0, 0);
	float choose = drand48();
	float w = 0.0;

	for(long unsigned int i = 0; i < command.size(); ++i)
	{
		w += weights[i];
		if(choose <= w)
		{
			u = command[i];
			break;
		} 
	}

	float f = u(0);
	float s = u(1);
	float r = u(2);

	float f_h = f - SampleGuassian(noise(0) * fabs(f));
	float s_h = s - SampleGuassian(noise(1) * fabs(s));
	float r_h = r - SampleGuassian(noise(2) * fabs(r));


	Eigen::Vector3f new_p = Forward(p1, Eigen::Vector3f(f_h, s_h, r_h));

	return new_p;

}


Eigen::Vector3f MixedFSR::Backward(Eigen::Vector3f p1, Eigen::Vector3f p2)
{
	Eigen::Vector3f dp = p2 - p1;

	float a = cos(p1(2));
	float b = sin(p1(2));
	
    float f = (dp.x() * a + dp.y() * b) /  (pow(a, 2.0) + pow(b, 2.0));
    float s = (dp.y() - f * b) / a;
    float r = dp(2);


	return Eigen::Vector3f(f, s, r);

}

Eigen::Vector3f MixedFSR::Forward(Eigen::Vector3f p1, Eigen::Vector3f u)
{
	float f = u(0);
	float s = u(1);
	float r = u(2);

	float x = p1(0) + f * cos(p1(2)) - s * sin(p1(2));
	float y = p1(1) + f * sin(p1(2)) + s * cos(p1(2));
	float theta = Wrap2Pi(r + p1(2));

	return Eigen::Vector3f(x, y, theta);

}



