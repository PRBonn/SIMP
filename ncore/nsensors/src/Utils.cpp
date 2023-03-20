/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: Utils.cpp                                                             #
# ##############################################################################
**/

#include <math.h>
#include "Utils.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>



Eigen::Matrix3f Vec2Trans(Eigen::Vector3f v)
{
	float c = cos(v(2));
	float s = sin(v(2));
	Eigen::Matrix3f trans;
	trans << c, -s, v(0),	s, c, v(1),	0, 0, 1;

	return trans;
}

float Wrap2Pi(float angle)
{
	float wAngle = angle;
	while (wAngle < -M_PI) wAngle += 2 * M_PI;

	while (wAngle > M_PI) wAngle -= 2 * M_PI;

	return wAngle;
}

float GetYaw(float qz, float qw)
{
	float yaw = 2 * atan2(qz, qw);
	yaw = Wrap2Pi(yaw);

	return yaw;
}

std::vector<Eigen::Vector3f> Ranges2Points(const std::vector<float>& ranges, const std::vector<float>& angles)
{
	int n = ranges.size();
	std::vector<Eigen::Vector3f> homoPoints(n);

	for(int i = 0; i < n; ++i)
	{
		float r = ranges[i];
		float a = angles[i];
		Eigen::Vector3f p = Eigen::Vector3f(r * cos(a), r * sin(a), 1);
		homoPoints[i] = p;
	}

	// consider using a matrix instead of a list for later stuff
	return homoPoints;
}



std::vector<float> Downsample(const std::vector<float>& ranges, int N)
{
	int n = ranges.size() / N;
	std::vector<float> downsampled(n);

	for(int i = 0; i < n; ++i)
	{
		downsampled[i] = ranges[i * N];
	}

	return downsampled;
}

std::vector<double> Downsample(const std::vector<double>& ranges, int N)
{
	int n = ranges.size() / N;
	std::vector<double> downsampled(n);

	for(int i = 0; i < n; ++i)
	{
		downsampled[i] = ranges[i * N];
	}

	return downsampled;
}



std::vector<float> StringToVec(std::string seq)
{
	std::vector<float> vec;

	seq.erase(std::remove(seq.begin(), seq.end(), ','), seq.end());
	seq.erase(std::remove(seq.begin(), seq.end(), '['), seq.end());
	seq.erase(std::remove(seq.begin(), seq.end(), ']'), seq.end());

	size_t pos = 0;
	std::string space_delimiter = " ";
	std::vector<std::string> words;
	while ((pos = seq.find(space_delimiter)) != std::string::npos) 
	{
        words.push_back(seq.substr(0, pos));
        seq.erase(0, pos + space_delimiter.length());
    }
    words.push_back(seq.substr(0, pos));

    for(long unsigned int i = 0; i < words.size(); ++i)
    {
    	//std::cout << words[i] << std::endl;
    	vec.push_back(std::stof(words[i]));
    }

    return vec;
}


std::vector<std::string> File2Lines(std::string filePath)
{
	std::ifstream file(filePath);
	
	std::vector<std::string> fields;

	if (file.is_open()) 
	{
	    std::string line;
	    while (std::getline(file, line)) 
	    {
	       fields.push_back(line);
	    }
	    file.close();
	}

	return fields;
}


float SampleGuassian(float sigma)
{
	float sample = 0;

	for(int i = 0; i < 12; ++i)
	{
		sample += drand48() * 2 * sigma - sigma;
	}
	sample *= 0.5;

	return sample;

}