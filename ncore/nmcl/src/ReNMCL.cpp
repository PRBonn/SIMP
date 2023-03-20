/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ReNMCL.cpp                    			                               #
# ##############################################################################
**/


#include "ReNMCL.h"
#include <numeric>
#include <functional> 
#include <iostream>
#include <fstream>

ReNMCL::ReNMCL(std::shared_ptr<BuildingMap> bm, std::shared_ptr<Resampling> rs, std::shared_ptr<MixedFSR> mm,
 std::shared_ptr<SemanticOverlap> sem3d, int n)
{
	o_motionModel = mm;
	
	o_resampler = rs;
	o_numParticles = n;
	o_buildingMap = bm;
	o_floorMap = o_buildingMap->GetFloors()[0];


	const std::vector<std::shared_ptr<FloorMap>>& floors = o_buildingMap->GetFloors();
	for (int i = 0; i < floors.size(); ++i)
	{
		o_gmaps.push_back(floors[i]->Map());
	}

	o_numFloors = floors.size();

	o_particleFilter = std::make_shared<ParticleFilter>(ParticleFilter(o_buildingMap));
	o_particleFilter->InitUniform(o_particles, o_numParticles);
	o_stats = SetStatistics::ComputeParticleSetStatistics(o_particles);
	o_semanticModel3 = sem3d;
	dumpParticles();

}

ReNMCL::ReNMCL(std::shared_ptr<BuildingMap> bm,  std::shared_ptr<Resampling> rs, std::shared_ptr<MixedFSR> mm, std::shared_ptr<SemanticOverlap> sem3d,
			int n, std::vector<Eigen::Vector4f> initGuess, std::vector<Eigen::Matrix3d> covariances)
{
	o_motionModel = mm;
	o_resampler = rs;
	o_numParticles = n;
	o_buildingMap = bm;
	o_floorMap = o_buildingMap->GetFloors()[0];

	const std::vector<std::shared_ptr<FloorMap>>& floors = o_buildingMap->GetFloors();
	for (int i = 0; i < floors.size(); ++i)
	{
		o_gmaps.push_back(floors[i]->Map());
	}

	o_numFloors = floors.size();
	
	o_particleFilter = std::make_shared<ParticleFilter>(ParticleFilter(o_buildingMap));
	o_particleFilter->InitGaussian(o_particles, o_numParticles, initGuess, covariances);
	o_stats = o_particleFilter->ComputeStatistics(o_particles);
	o_semanticModel3 = sem3d;
	dumpParticles();
}


void ReNMCL::CorrectOmni(std::shared_ptr<Semantic3DData> data)
{
	o_semanticModel3->ComputeWeights(o_particles, data);
	o_particleFilter->NormalizeWeights(o_particles);
	o_resampler->Resample(o_particles);
	o_stats = o_particleFilter->ComputeStatistics(o_particles);
}

void ReNMCL::dumpParticles()
{
	std::string path = "PMap/" + std::to_string(o_frame) + "_particles.csv";
	std::ofstream particleFile;
    particleFile.open(path, std::ofstream::out);
    particleFile << "x" << "," << "y" << "," << "yaw" << "," << "id" << "," << "w" << std::endl;
    for(int p = 0; p < o_particles.size(); ++p)
    {
        Eigen::Vector3f pose = o_particles[p].pose;
        float w = o_particles[p].weight;
        unsigned int floorID = o_particles[p].floorID;
        particleFile << pose(0) << "," << pose(1) << "," << pose(2) << "," << floorID << "," << w << std::endl;
    }
    particleFile.close();
    ++o_frame;
}


void ReNMCL::Predict(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	predictUniform(u, odomWeights, noise);
}

void ReNMCL::predictUniform(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	for(int i = 0; i < o_numParticles; ++i)
	{
		Eigen::Vector3f pose = o_motionModel->SampleMotion(o_particles[i].pose, u, odomWeights, noise);
		o_particles[i].pose = pose;
		unsigned int floorID = o_particles[i].floorID;
		// if (drand48() < 0.05)
		// {
		// 	floorID = floor(o_numFloors * drand48());
		// }
	
		//particle pruning - if particle is outside the map, we replace it
		while (!o_gmaps[floorID]->IsValid(o_particles[i].pose))
		{
			std::vector<Particle> new_particle;
			o_particleFilter->InitUniform(new_particle, 1);
			new_particle[0].weight = 1.0 / o_numParticles;
			new_particle[0].floorID = floorID;
			o_particles[i] = new_particle[0];
		}
	}
}



void ReNMCL::Recover()
{
	o_particleFilter->InitUniform(o_particles, o_numParticles);
}





