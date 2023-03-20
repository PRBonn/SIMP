/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: Semantic3DData.h          		                           		       #
# ##############################################################################
**/


#ifndef SEMANTIC3DDATA_H
#define SEMANTIC3DDATA_H

#include <vector>
#include <eigen3/Eigen/Dense>


class Semantic3DData 
{

	public: 

		Semantic3DData(const std::vector<int>&  labels, const std::vector<std::vector<Eigen::Vector3f>>& vertices, const std::vector<float>& confidences)
		{
			o_labels = labels;
			o_vertices = vertices;
			o_confidences = confidences;
		}

		//virtual ~LidarData(){};

		const std::vector<std::vector<Eigen::Vector3f>>& Vertices() const
		{
			return o_vertices;
		}

		const std::vector<int>& Label() const
		{
			return o_labels;
		}

		const std::vector<float>& Confidence() const
		{
			return o_confidences;
		}

	private:

		std::vector<std::vector<Eigen::Vector3f>> o_vertices;
		std::vector<int> o_labels;
		std::vector<float> o_confidences;

};

#endif