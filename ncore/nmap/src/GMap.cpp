/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: GMap.cpp                                                              #
# ##############################################################################
**/

#include "GMap.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include "Utils.h"

GMap::GMap(cv::Mat& gridMap, Eigen::Vector3f origin, float resolution)
{
	o_resolution = resolution;
	o_origin = origin;
	if (gridMap.channels() == 3)
	{
		cv::cvtColor(gridMap, o_gridmap, cv::COLOR_BGR2GRAY);
	} 
	else
	{
		gridMap.copyTo(o_gridmap);
	}

	o_gridmap = 255 - o_gridmap;
	o_maxy = o_gridmap.rows;

	//compute the borders	
	getBorders();
}


GMap::GMap(const std::string& mapFolder, const std::string& yamlName)
{
	
	std::vector<std::string> fields = File2Lines(mapFolder + yamlName);

	// "image: " - 7
	fields[0].erase(0,7);
	// "iresolution: " - 12
	fields[1].erase(0,12);
	// "origin: " - 8
	fields[2].erase(0,8);
	// "occupied_thresh: " - 17
	fields[4].erase(0,17);
	// "free_thresh: " - 13
	fields[5].erase(0,13);

	std::string imgPath = mapFolder + fields[0];
	o_resolution = std::stof(fields[1]);

	std::vector<float> vec = StringToVec(fields[2]);

	o_origin = Eigen::Vector3f(vec[0], vec[1], vec[2]);
	o_gridmap = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
	//cv::cvtColor( gridMap_, map, cv::COLOR_BGR2GRAY);
	o_gridmap = 255 - o_gridmap;
	o_maxy = o_gridmap.rows;

	//compute the borders	
	getBorders();
}


void GMap::getBorders()
{
	cv::Mat binMap;
	cv::threshold(o_gridmap, binMap, 127, 255, 0);
	std::vector<cv::Point> locations; 
	cv::findNonZero(binMap, locations);
	cv::Rect rect = cv::boundingRect(locations);

	// remember: cv::Point = (col, row) - might consider inverting this
	// or using Eigen
	cv::Point tl = rect.tl();
	cv::Point br = rect.br();
	o_topLeft = Eigen::Vector2f(tl.x, tl.y);
	o_bottomRight = Eigen::Vector2f((br.x < o_gridmap.cols) ? br.x : o_gridmap.cols - 1, (br.y < o_maxy) ? br.y : (o_maxy - 1));
} 

Eigen::Vector2f GMap::World2Map(Eigen::Vector2f xy) const
{
	int u = round((xy(0) - o_origin(0)) / o_resolution);
	int v = o_maxy - round((xy(1) - o_origin(1)) / o_resolution);
	return Eigen::Vector2f(u, v);
}



Eigen::Vector2f GMap::Map2World(Eigen::Vector2f uv) const
{
	float x = uv(0) * o_resolution + o_origin(0);
	float y = (o_maxy - uv(1)) * o_resolution + o_origin(1);
	return Eigen::Vector2f(x, y);
}


bool GMap::IsValid(Eigen::Vector3f pose) const
{
	Eigen::Vector2f xy = Eigen::Vector2f(pose(0), pose(1));
	Eigen::Vector2f mp = World2Map(xy);

	return IsValid2D(mp);
}


bool GMap::IsValid2D(Eigen::Vector2f mp) const
{
	Eigen::Vector2f br = o_bottomRight;
	if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1))) return false;

	int val = o_gridmap.at<uchar>(mp(1) ,mp(0));

	if (val > 1) return false;

	return true;
}