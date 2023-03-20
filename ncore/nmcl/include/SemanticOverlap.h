/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SemanticOverlap.h                      		                       #
# ##############################################################################
**/


#ifndef SEMANTICOVERLAP_H
#define SEMANTICOVERLAP_H

#include <memory>
#include <string>
#include <Particle.h>
#include "GMap.h"
#include "BuildingMap.h"
#include "Semantic3DData.h"


class SemanticOverlap
{
	public:



		//! A constructor
	    /*!
	      \param buildingMap is a ptr to a Building object, which holds the buiding map
	      \param classes is a float that determine how forgiving the model is (small sigma will give a very peaked likelihood)
	      \param confidences is a float specifying up to what distance from the sensor a reading is valid
		  \param globalVariancevPath is a float specifying up to what distance from the sensor a reading is valid
	    */

		SemanticOverlap(std::shared_ptr<BuildingMap> buildingMap, const std::vector<std::string>& classes, const std::vector<float>& confidences, const std::string& globalVariancevPath);


			//! Computes weights for all particles based on how well the observation matches the map
		/*!
		  \param particles is a vector of Particle elements
		  \param SensorData is an abstract container for sensor data. This function expects LidarData type
		*/

		void ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<Semantic3DData> data);

	

	private:

	
		bool isTraced(const cv::Mat& currMap, Eigen::Vector2f pose, Eigen::Vector2f bearing, unsigned int floorID);

		double geometric(const Particle& particle, std::shared_ptr<Semantic3DData> data);
		std::pair <int,float>  computeLikelihood(Eigen::Vector2f uv, int label,  int floorID);
		void createObjectMaps();


		//void createVisibilityMap(unsigned int floorID);

		std::vector<std::string> o_classNames;
		std::vector<std::shared_ptr<GMap>> o_gmaps;
		cv::Size o_mapSize;
		std::vector<float> o_confidenceTH;
		std::vector<float> o_sigmas;
		std::vector<Eigen::Vector2f> o_classConsistency;
		std::shared_ptr<BuildingMap> o_buildingMap;
		float o_maxRange = 10.0;
 
		std::vector<int> o_categories;
		std::vector<int> o_uid;
		std::vector<float> o_mx;
		std::vector<float> o_my;
		std::vector<float> o_sx;
		std::vector<float> o_sy;
		std::vector<float> o_b;
		std::vector<std::vector<cv::Point>>  o_corners;
		std::vector<cv::Mat> o_objectMaps;
		
};





#endif
