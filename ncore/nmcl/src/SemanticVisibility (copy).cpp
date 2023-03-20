/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: SemanticVisibility.cpp                                                #
# ##############################################################################
**/


#include "SemanticVisibility.h"
#include "Utils.h"
#include "math.h"

SemanticVisibility::SemanticVisibility(std::shared_ptr<BuildingMap> buildingMap, int beams, const std::vector<std::string>& classNames, const std::vector<float>& confidences)
{
	o_buildingMap = buildingMap;
	const std::vector<std::shared_ptr<FloorMap>>& floors = o_buildingMap->GetFloors();
	o_numBeams = beams;
	o_classNames = classNames;
	o_confidenceTH = confidences;

	o_visibilityMaps = std::vector<VisibilityMap>(floors.size());

	o_mapSize = floors[0]->Map()->Map().size();

	for (int i = 0; i < floors.size(); ++i)
	{
		o_gmaps.push_back(floors[i]->Map());
		createVisibilityMap(i);
	}

	//o_mapSize = o_gmaps[0]->Map().size();

}

void SemanticVisibility::createVisibilityMap(unsigned int floorID)
{

	VisibilityMap visibilityMap = VisibilityMap(o_mapSize.width * o_mapSize.height);

	std::vector<Eigen::Vector2f> unitCircle(o_numBeams);
	for(int i = 0; i < o_numBeams; ++i)
	{
		float angle = 2.0 * i * M_PI / float(o_numBeams);
		unitCircle[i] = Eigen::Vector2f(cos(angle), sin(angle));
		//std::cout << unitCircle[i](0) << ", " << unitCircle[i](1) << std::endl;
	}


	std::vector<cv::Mat> classMaps = o_buildingMap->GetFloors()[floorID]->CreateSemMaps();

#ifdef DEBUG
	std::vector<cv::Mat> debugMaps;
	for(int i = 0; i < o_classNames.size(); ++i)
	{
		cv::Mat debug(o_mapSize, CV_8U, cv::Scalar(0)); 
		debugMaps.push_back(debug);
	}
#endif

	o_classConsistency = std::vector<Eigen::Vector2f>(o_classNames.size(), Eigen::Vector2f(0, 0));

	//for each free (!) image pixel 
	#pragma omp parallel for 
	for (long unsigned int row = 0; row < o_mapSize.height; ++row)
	{
		for (long unsigned int col = 0; col < o_mapSize.width; ++col)
		{
			Eigen::Vector2f pose(col, row);
			
			std::map<int, std::vector<Eigen::Vector2f>> cell = std::map<int, std::vector<Eigen::Vector2f>>();

			if (o_gmaps[floorID]->IsValid2D(pose)) 
			{
				for (long unsigned int c = 0; c < classMaps.size(); ++c)
				{
					std::vector<Eigen::Vector2f> brs;
					// load map for class c
					cv::Mat currMap = classMaps[c];

					// ray trace in ~20 directions - check map occupied, check hitting object, check in the map
					for(long unsigned int u = 0; u < unitCircle.size(); u++)
					{
						Eigen::Vector2f bearing = unitCircle[u];
						bool traced = isTraced(currMap, pose, bearing, floorID);
						if (traced)
						{
							brs.push_back(bearing);
#ifdef DEBUG
							debugMaps[c].at<uchar>(row, col) = 255;
								
#endif
						}
					}
					// add to DB
					if (brs.size()) cell[c] = brs;
				}
			}
			visibilityMap[cellID(col, row)] = cell;
		}
	}
#ifdef DEBUG

	for(int i = 0; i < o_classNames.size(); ++i)
	{
		cv::imwrite("f" + std::to_string(floorID) + "_" + o_classNames[i] + "_debug.png", debugMaps[i]);
	}

#endif	

	o_visibilityMaps[floorID] = visibilityMap;
}



bool SemanticVisibility::isTraced(const cv::Mat& currMap, Eigen::Vector2f pose, Eigen::Vector2f bearing, unsigned int floorID)
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


void SemanticVisibility::ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<SemanticData> data)
{
	const std::vector<Eigen::Vector2f>& poses = data->Pos();
	const std::vector<int>& labels = data->Label();
	const std::vector<float>& confidences = data->Confidence();


	for(long unsigned int p = 0; p < particles.size(); ++p)
	{
		Eigen::Vector3f pose = particles[p].pose;
		unsigned int floorID = particles[p].floorID;
		Eigen::Vector2f br = o_gmaps[floorID]->BottomRight();

		Eigen::Vector2f xy = Eigen::Vector2f(pose(0), pose(1));
		Eigen::Vector2f mp = o_gmaps[floorID]->World2Map(xy);
		Eigen::Matrix3f trans = Vec2Trans(pose);

		float w = 1.0;
		float dist = 0.0;

		if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
		{
				w = 0.0;
		}
		else
		{
			int cID = cellID(mp(0), mp(1));
			std::map<int, std::vector<Eigen::Vector2f>> cell = o_visibilityMaps[floorID][cID];

			for (long unsigned int d = 0; d < labels.size(); ++d)
			{
				int label = labels[d];
				float conf = confidences[d];
				if (conf < o_confidenceTH[label])
				{
					//w *= 0.1;
					//dist += 0.9;
					continue;
				}

				Eigen::Vector2f pr_pose = poses[d];
				// convert from base_link to 3D world frame, flip y axis
				//Eigen::Vector3f ts = trans * Eigen::Vector3f(pr_pose(0), -pr_pose(1), 1);
				Eigen::Vector3f ts = trans * Eigen::Vector3f(pr_pose(0), pr_pose(1), 1);
				// project onto the map in 2D
				Eigen::Vector2f pr_uv = o_gmaps[floorID]->World2Map(Eigen::Vector2f(ts(0), ts(1)));
				Eigen::Vector2f pr_bearing = (pr_uv - mp).normalized();

				std::map<int, std::vector<Eigen::Vector2f>>::iterator it = cell.find(label);
				if (it != cell.end())
				{
					std::vector<Eigen::Vector2f> mp_bearings = cell[label];
					std::vector<float> dot_scores;
					
					for(long unsigned int b = 0; b < mp_bearings.size(); ++b)
					{
						Eigen::Vector2f mp_bearing =  mp_bearings[b];
						float score = mp_bearing.dot(pr_bearing);
						dot_scores.push_back(score);
					}

					//does dor product return a number between -1 and 1? verify!!
					float max_score = *max_element(dot_scores.begin(), dot_scores.end());
					max_score = 0.5 * (max_score + 1.0);
					//w *= max_score;
					dist += (1 - max_score);  
				}
				else
				{
					// no object of this class found in the map
					// down-weight particles
					//w *= 0.001;
					dist += 10 ;
				}
			}
			w = exp(-(dist)/ labels.size());
		}

		particles[p].weight = w;

	}

}


void SemanticVisibility::UpdateConsistency(const Particle& particle, std::shared_ptr<SemanticData> data)
{
	const std::vector<Eigen::Vector2f>& poses = data->Pos();
	const std::vector<int>& labels = data->Label();
	const std::vector<float>& confidences = data->Confidence();
	unsigned int floorID = particle.floorID;
	Eigen::Vector2f br = o_gmaps[floorID]->BottomRight();

	Eigen::Vector3f pose = particle.pose;
	Eigen::Vector2f xy = Eigen::Vector2f(pose(0), pose(1));
	Eigen::Vector2f mp = o_gmaps[floorID]->World2Map(xy);
	Eigen::Matrix3f trans = Vec2Trans(pose);

	if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
	{
			return;
	}
	else
	{
		int cID = cellID(mp(0), mp(1));
		std::map<int, std::vector<Eigen::Vector2f>> cell = o_visibilityMaps[floorID][cID];

		for (long unsigned int d = 0; d < labels.size(); ++d)
		{
			int label = labels[d];
			float conf = confidences[d];
			if (conf < o_confidenceTH[label])
			{
				continue;
			}

			Eigen::Vector2f pr_pose = poses[d];
			// convert from base_link to 3D world frame, flip y axis
			//Eigen::Vector3f ts = trans * Eigen::Vector3f(pr_pose(0), -pr_pose(1), 1);
			Eigen::Vector3f ts = trans * Eigen::Vector3f(pr_pose(0), pr_pose(1), 1);
			// project onto the map in 2D
			Eigen::Vector2f pr_uv = o_gmaps[floorID]->World2Map(Eigen::Vector2f(ts(0), ts(1)));
			Eigen::Vector2f pr_bearing = (pr_uv - mp).normalized();

			std::map<int, std::vector<Eigen::Vector2f>>::iterator it = cell.find(label);
			if (it != cell.end())
			{
				std::vector<Eigen::Vector2f> mp_bearings = cell[label];
				std::vector<float> dot_scores;
				
				for(long unsigned int b = 0; b < mp_bearings.size(); ++b)
				{
					Eigen::Vector2f mp_bearing =  mp_bearings[b];
					float score = mp_bearing.dot(pr_bearing);
					dot_scores.push_back(score);
				}

				//does dor product return a number between -1 and 1? verify!!
				float max_score = *max_element(dot_scores.begin(), dot_scores.end());
				max_score = 0.5 * (max_score + 1.0);
				
				if(max_score > 0.95)
				{
					o_classConsistency[label](0) += 1.0;
				}
			}
			o_classConsistency[label](1) += 1.0;
		}
	}

}





int SemanticVisibility::cellID(int x, int y)
{

	return y * o_mapSize.width + x;
}