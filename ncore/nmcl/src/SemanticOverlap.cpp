/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SemanticOverlap.cpp                      		                       #
# ##############################################################################
**/

#include "SemanticOverlap.h"
#include "Utils.h"
#include "math.h"
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

SemanticOverlap::SemanticOverlap(std::shared_ptr<BuildingMap> buildingMap, const std::vector<std::string>& classNames, const std::vector<float>& confidences, const std::string& globalVariancevPath)
{
	o_buildingMap = buildingMap;
	const std::vector<std::shared_ptr<FloorMap>>& floors = o_buildingMap->GetFloors();
	o_classNames = classNames;
	o_confidenceTH = confidences;

	o_mapSize = floors[0]->Map()->Map().size();
	o_gmaps.push_back(floors[0]->Map());
	
	std::ifstream file(globalVariancevPath);
	
	json config;
	file >> config;
	std::vector<int> categories = config["category"];
	std::vector<int>  uid = config["uid"];
	std::vector<float>  mx = config["mx"];
	std::vector<float>  my = config["my"];
	std::vector<float>  sx = config["sx"];
	std::vector<float>  sy = config["sy"];
	std::vector<float>  b = config["b"];
	o_categories = categories;
	o_uid = uid;
	o_mx = mx;
	o_my = my;
	o_sx = sx;
	o_sy = sy;
	o_b = b;
	std::vector<std::vector<int>>  corners = config["corners"];
	o_corners = std::vector<std::vector<cv::Point>>(corners.size(), std::vector<cv::Point>());
	for (int i = 0; i < corners.size(); ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			o_corners[i].push_back(cv::Point(corners[i][2 * j], corners[i][2 * j + 1]));
		}
	}

	createObjectMaps();
} 

void SemanticOverlap::createObjectMaps()
{
	for (int i = 0; i < o_corners.size(); ++i)
	{
		cv::Mat objMap = cv::Mat::zeros(o_mapSize, CV_8U);
		std::vector<cv::Point> hull;
		cv::convexHull( o_corners[i], hull );
		// create binary mask
		cv::fillConvexPoly(objMap, hull.data(), 4, cv::Scalar(255));
		o_objectMaps.push_back(objMap);
	}
}

bool SemanticOverlap::isTraced(const cv::Mat& currMap, Eigen::Vector2f pose, Eigen::Vector2f bearing, unsigned int floorID)
{
	Eigen::Vector2f currPose = pose;
	float step = 1;

	while(o_gmaps[floorID]->IsValid2D(currPose))
	{
		currPose += step * bearing;
		if(currMap.at<uchar>(currPose(1), currPose(0)))
		{
			return true;
		}
	}
	return false;
}



void SemanticOverlap::ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<Semantic3DData> data)
{
	//#pragma omp parallel for 
	for(long unsigned int p = 0; p < particles.size(); ++p)
	{
		double w =  geometric(particles[p], data);
		particles[p].weight = w;
		//if (std::isnan(w)) std::cout << "nan weight" << std::endl;
		//std::cout << w << std::endl;
	}
}


/* Thanks Meta for the nice drawing of how vertices are organized and how we define vertIDs
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
*/


//#define DEBUG


std::pair <int,float> SemanticOverlap::computeLikelihood(Eigen::Vector2f uv, int label, int floorID)
{
		int id = 0;
	
		// take max of gaussian mixture
		float l = FLT_MAX;
		for(int i = 0; i < o_uid.size(); ++i)
		{
			if (o_categories[i] == label)
			{
				int b = o_b[i];
				float mx = o_mx[i];
				float my = o_my[i];
				float sx = o_sx[i];
				float sy = o_sy[i];
				float tmp = (pow(((uv[0] - mx) / sx ), b) + pow(((uv[1] - my) / sy ), b)) ;

				if (tmp < l)
				{
					l = tmp;
					id = i;
				}
			}
		}
		if (l > o_maxRange) l = o_maxRange;

		l = exp(-l);

		return std::make_pair(id, l);
}


double SemanticOverlap::geometric(const Particle& particle, std::shared_ptr<Semantic3DData> data)
{
	const std::vector<std::vector<Eigen::Vector3f>>& vertices = data->Vertices();
	const std::vector<int>& labels = data->Label();
	const std::vector<float>& confidences = data->Confidence();

	Eigen::Vector3f pose = particle.pose;
	unsigned int floorID = particle.floorID;
	Eigen::Vector2f br = o_gmaps[floorID]->BottomRight();

	Eigen::Vector2f xy = Eigen::Vector2f(pose(0), pose(1));
	Eigen::Vector2f mp = o_gmaps[floorID]->World2Map(xy);
	Eigen::Matrix3f trans = Vec2Trans(pose);

	float w = 1.0;
	double likelihood = 0.0;
	int valid = 0;

	float dist = 0.0;
	double totOverlapScore = 0.0;
	int penalty = o_maxRange;
	float sigma = 4;
	double p_center = 1.0;
	double p_not_center = 1.0;
	double p_uniform = 1.0;
	double cyrill_cnst = 0.00001;

	int vertIDs[4] = {0, 1, 2, 3};

	//std::cout << "start sensor model" << std::endl;
	if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
	{
			w = 0;
	}
	else
	{
		for (long unsigned int d = 0; d < labels.size(); ++d)
		{
			int label = labels[d];
			float conf = confidences[d];

			if (conf < o_confidenceTH[label])
			{
				continue;
			}

			std::vector<Eigen::Vector3f> cube = vertices[d];
			sigma = o_sigmas[label];

			std::vector<cv::Point> points = std::vector<cv::Point>(4);
			bool inMap = false;
			Eigen::Vector2f uv_mean = Eigen::Vector2f(0.0, 0.0);
			for(int b = 0; b < 4; ++b)
			{
				Eigen::Vector3f vert3d = cube[vertIDs[b]];
				Eigen::Vector3f ts = trans * Eigen::Vector3f(vert3d(0), vert3d(1), 1);
				Eigen::Vector2f uv = o_gmaps[floorID]->World2Map(Eigen::Vector2f(ts(0), ts(1)));
				uv_mean += uv;
				points[b] = cv::Point(uv(0), uv(1));

				if  ((uv(0) >= 0) and (uv(0) < o_mapSize.width) and (uv(1) >= 0) and (uv(1) < o_mapSize.height))
				{
					inMap = true;
				}	
			}

			if (inMap == false)
			{
				totOverlapScore += 2 * penalty;
				p_center *= 0.5;
				p_not_center *= 0.5;
				p_uniform *= cyrill_cnst;
				valid = true;
			 	continue;
			}

			uv_mean = uv_mean / 4;
			std::pair<int, float> res = computeLikelihood(uv_mean, label, floorID);	
			double likelihood = res.second;
			int id = res.first;

			 std::vector<cv::Point> hull;
			 cv::convexHull( points, hull );
			// create rotated rectangle
			cv::RotatedRect rotRect = cv::minAreaRect(points);
			// create binary mask
			cv::Mat binMask = cv::Mat::zeros(o_mapSize, CV_8U);
			cv::fillConvexPoly(binMask, hull.data(), 4, cv::Scalar(255));
			// get the axis-aligned rectangle and crop patch
			cv::Rect rect = rotRect.boundingRect();
			// check rect is not outside image
			if (rect.x < 0) rect.x = 0;
			if (rect.x >= binMask.cols) rect.x = binMask.cols-1;
			if (rect.width + rect.x >= binMask.cols)
			{
				rect.width = binMask.cols - rect.x -1;
			}
			if (rect.y < 0) rect.y = 0;
			if (rect.y >= binMask.rows) rect.y = binMask.rows-1;
			if (rect.height + rect.y >= binMask.rows)
			{
				rect.height = binMask.rows - rect.y -1;
			}
			cv::Mat rectPatch = binMask(rect);	
			double areaSum = cv::sum(rectPatch)[0];
			if (areaSum == 0)
			{
				totOverlapScore += penalty ;
				p_center *= 0.5;
				p_not_center *= 0.5;
				p_uniform *= cyrill_cnst;
				valid = true;
			 	continue;
			}
			
			// get the corresponding patch from the semantic map			
			cv::Mat mapPatch = o_objectMaps[id](rect);		

			//compute the AND and divide by the rotatedRectangle area to normalize get overlap
			cv::Mat overlapMat;
			cv::bitwise_and(rectPatch, mapPatch, overlapMat);
			double overlapSum = cv::sum(overlapMat)[0];

			cv::Mat unionMat;
			cv::bitwise_or(rectPatch, mapPatch, unionMat);
			double unionSum = cv::sum(unionMat)[0];

			double overlap_area = 0.0;
			if (unionSum)
			{
				overlap_area = overlapSum / unionSum;
			}
			

#ifdef DEBUG 
			cv::Mat gmap = o_gmaps[floorID]->Map();
			std::vector<cv::Mat> channels = {binMask, gmap, mapMask};
			cv::Mat merge;
			cv::merge(channels, merge);
			cv::circle( merge, cv::Point(mp(0), mp(1)), 4, cv::Scalar( 255, 255, 255), cv::FILLED, cv::LINE_8 );
			cv::imshow("merge", merge);
			cv::waitKey();
#endif

			if (o_objectMaps[id].at<uchar>(uv_mean(1) ,uv_mean(0))) likelihood = 0.99;
			double overlapScore = 1.0 - overlap_area;
			totOverlapScore += overlapScore;
			p_center *= likelihood;
			p_not_center *= (1.0 - likelihood);
			p_uniform *= cyrill_cnst;
			valid = true;

		}

		//w = exp(-totOverlapScore / labels.size());
		double w_all = exp(-pow(totOverlapScore, 2.0 )) * p_center  + p_not_center * p_uniform;
		w = pow( w_all, 1.0 / labels.size());
		//valid += 1;
	}

	//std::cout << w << std::endl;
	if (valid ) return w;
	return particle.weight;
}


