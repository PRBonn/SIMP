/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: Lidar2D.cpp                                                           #
# ##############################################################################
**/


#include "Lidar2D.h"
#include "Utils.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>

Lidar2D::Lidar2D(std::string name, Eigen::Vector3f origin, int numBeams, float maxAngle, float minAngle)
{
	o_name = name;
	o_trans = Vec2Trans(origin);
	

	float reso = (maxAngle - minAngle) / (numBeams - 1);
	for(int i = 0; i < numBeams; ++i)
	{
		float angle =  minAngle + i * reso;
		o_heading.push_back(angle);
	}
}

Lidar2D::Lidar2D(std::string jsonPath)
{
	using json = nlohmann::json;

	std::ifstream file(jsonPath);
	json config;
	file >> config;
	o_name = config["name"];
	float maxAngle = config["angle_max"];
	float minAngle = config["angle_min"];
	int numBeams = config["num_beams"];
	std::vector<float> origin = config["origin"];
	o_trans = Vec2Trans(Eigen::Vector3f(origin[0], origin[1], origin[2]));
	float reso = (maxAngle - minAngle) / (numBeams - 1);
	for(int i = 0; i < numBeams; ++i)
	{
		float angle =  minAngle + i * reso;
		o_heading.push_back(angle);
	}
	
}


Lidar2D::Lidar2D(std::string name, std::string yamlFolder)
{
	o_name = name;
	std::vector<std::string> fields = File2Lines(yamlFolder + name + ".yaml");

	// "angle_max: " - 11
	fields[1].erase(0,11);
	// "angle_min: " - 11
	fields[2].erase(0,11);
	// "num_beams: " - 11
	fields[3].erase(0,11);
	// "origin: " - 8
	fields[4].erase(0,8);

	float maxAngle = std::stof(fields[1]);
	float minAngle = std::stof(fields[2]);
	int numBeams = std::stoi(fields[3]);
	std::vector<float> vec = StringToVec(fields[4]);
	Eigen::Vector3f origin = Eigen::Vector3f(vec[0], vec[1], vec[2]);
	o_trans = Vec2Trans(origin);

	float reso = (maxAngle - minAngle) / (numBeams - 1);
	for(int i = 0; i < numBeams; ++i)
	{
		float angle =  minAngle + i * reso;
		o_heading.push_back(angle);
	}
}


std::vector<Eigen::Vector3f> Lidar2D::Center(std::vector<Eigen::Vector3f>& homoPoints)
{
	int n = homoPoints.size(); 
	std::vector<Eigen::Vector3f> transPoints(n);

	for(int i = 0; i < n; i++)
	{
		Eigen::Vector3f p = homoPoints[i];
		Eigen::Vector3f p_trans = o_trans * p;
		transPoints[i] = p_trans;
	}

	return transPoints;
}

std::vector<Eigen::Vector3f> MergeScans(const std::vector<float>& f_ranges, Lidar2D laser_front, 
	const std::vector<float>& r_ranges, Lidar2D laser_rear, int dsFactor, float maxRange)
{
	
	int N = dsFactor;

	// front laser
	std::vector<float> f_ranges_sampled = Downsample(f_ranges, N);
	std::vector<float> f_angles = laser_front.Heading();
	std::vector<float> f_angles_sampled = Downsample(f_angles, N);

	auto idx = std::remove_if(f_ranges_sampled.begin(), f_ranges_sampled.end(), [=](float n) { return (n > maxRange || n < 0.0); });
	f_ranges_sampled.erase(idx, f_ranges_sampled.end());
	f_angles_sampled.erase(idx, f_angles_sampled.end());
	std::vector<Eigen::Vector3f> f_points_3d = Ranges2Points(f_ranges_sampled, f_angles_sampled);
	f_points_3d = laser_front.Center(f_points_3d);


	// rear laser
	std::vector<float> r_ranges_sampled = Downsample(r_ranges, N);
	std::vector<float> r_angles = laser_rear.Heading();
	std::vector<float> r_angles_sampled = Downsample(r_angles, N);

	idx = std::remove_if(r_ranges_sampled.begin(), r_ranges_sampled.end(), [maxRange](float n) { return (n > maxRange || n < 0.0); });
	r_ranges_sampled.erase(idx, r_ranges_sampled.end());
	r_angles_sampled.erase(idx, r_angles_sampled.end());
	std::vector<Eigen::Vector3f> r_points_3d = Ranges2Points(r_ranges_sampled, r_angles_sampled);
	r_points_3d = laser_rear.Center(r_points_3d);
	
	// combine scans

	f_points_3d.insert(f_points_3d.end(), r_points_3d.begin(), r_points_3d.end());

	return f_points_3d;
}

std::vector<Eigen::Vector3f> MergeScansSimple(const std::vector<float>& f_ranges, Lidar2D laser_front, 
	const std::vector<float>& r_ranges, Lidar2D laser_rear)
{

	std::vector<float> f_angles = laser_front.Heading();
	std::vector<Eigen::Vector3f> f_points_3d = Ranges2Points(f_ranges, f_angles);
	f_points_3d = laser_front.Center(f_points_3d);

	std::vector<float> r_angles = laser_rear.Heading();
	std::vector<Eigen::Vector3f> r_points_3d = Ranges2Points(r_ranges, r_angles);
	r_points_3d = laser_rear.Center(r_points_3d);

	f_points_3d.insert(f_points_3d.end(), r_points_3d.begin(), r_points_3d.end());

	return f_points_3d;
}