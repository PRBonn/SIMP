/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ParticleFilter.h             		                           		   #
# ##############################################################################
**/
 

#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include "Particle.h"
#include "GMap.h"
#include "SetStatistics.h"
#include "FloorMap.h"
#include "BuildingMap.h"

class ParticleFilter
{
public:

	//multi-map 
	ParticleFilter(std::shared_ptr<BuildingMap> buildingMap);

	void InitByRoomType(std::vector<Particle>& particles, int n_particles, const std::vector<float>& roomProbabilities);

	//multi-map 
	void InitUniform(std::vector<Particle>& particles, int n_particles);
    //multi-map 
	void InitGaussian(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector4f>& initGuess, const std::vector<Eigen::Matrix3d>& covariances);
    //multi-map
	void RemoveWeakest(std::vector<Particle>& particles, int n_particles);


	void AddBoundingBox(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector2f>& tls, const std::vector<Eigen::Vector2f>& brs, const std::vector<float>& yaws);

	//multi-map 
	SetStatistics ComputeStatistics(const std::vector<Particle>& particles);

	//multi-map 
	void NormalizeWeights(std::vector<Particle>& particles);

	//multi-map 
	std::vector<Particle> GetFloorParticles(const std::vector<Particle>& particles, unsigned int floorID);

	std::vector<Particle>& Particles()
	{
		return o_particles;
	}

	void SetParticle(int id, Particle p)
	{
		o_particles[id] = p;
	}


	SetStatistics Statistics()
	{
		return o_stats;
	}



private:



	std::shared_ptr<BuildingMap> o_buildingMap;
	std::shared_ptr<FloorMap> o_floorMap;
	std::shared_ptr<GMap> o_gmap;
	std::vector<std::shared_ptr<GMap>> o_gmaps;
	std::vector<Particle> o_particles;
	std::vector<SetStatistics> o_floorStats;
	SetStatistics o_stats;
	std::vector<double> o_floorWeights;
};

#endif