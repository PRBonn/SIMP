/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: GMap.h                                                                #
# ##############################################################################
**/


#ifndef GMAP
#define GMAP

#include "opencv2/opencv.hpp"
#include <memory>
#include "IMap2D.h"

class GMap : public IMap2D
{
	public:

		//! A constructor for handling the output of Gmapping, which include a metadata yaml and a .pgm map 
	    /*!
	      \param origin is the 2D pose of the bottom right corner of the map (found in the yaml)
	      \param resolution is the map resolution - distance in meters corresponding to 1 pixel (found in the yaml)
	      \param gridMap is occupancy map built according to the scans
	    */
		GMap(cv::Mat& gridMap, Eigen::Vector3f origin, float resolution);


		//! A constructor for handling the output of Gmapping, which include a metadata yaml and a .pgm map 
	    /*!
	      \param yamlPath is the path to the metadata yaml file produced by gmapping
	    */
		GMap(const std::string& mapFolder, const std::string& yamlName = "YouBotMap.yaml");
		

		//! A getter top left corner of the actual occupied area, as the map usually has wide empty margins 
		/*!
		   \return Eigen::Vector2f = (u, v) pixel coordinates for the gridmap
		*/
		Eigen::Vector2f TopLeft()
		{
			return o_topLeft;
		}


		//! A getter bottom right corner of the actual occupied area, as the map usually has wide empty margins 
		/*!
		   \return Eigen::Vector2f = (u, v) pixel coordinates for the gridmap
		*/
		Eigen::Vector2f BottomRight()
		{
			return o_bottomRight;
		}


		//! Converts (x, y) from the map frame to the pixels coordinates
		/*!
			\param (x, y) position in map frame
		   \return (u, v) pixel coordinates for the gridmap
		*/
		Eigen::Vector2f World2Map(Eigen::Vector2f xy) const;


		//! Converts (u, v) pixel coordinates to map frame (x, y)
		/*!
			\param (u, v) pixel coordinates for the gridmap
		   \return (x, y) position in map frame
		*/
		Eigen::Vector2f Map2World(Eigen::Vector2f uv) const;


		bool IsValid(Eigen::Vector3f pose) const;

		bool IsValid2D(Eigen::Vector2f mp) const;

		
		const cv::Mat& Map() const
		{
			return o_gridmap;
		}


	private:

		//std::shared_ptr<cv::Mat> o_gridmap;
		cv::Mat o_gridmap;
		float o_resolution = 0;
		int o_maxy = 0;
		Eigen::Vector3f o_origin;
		Eigen::Vector2f o_bottomRight;
		Eigen::Vector2f o_topLeft;

		void getBorders();
		
};

#endif //GMap

