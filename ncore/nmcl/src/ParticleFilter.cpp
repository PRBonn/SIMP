/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ParticleFilter.cpp           		                           		   #
# ##############################################################################
**/
 
 #include "ParticleFilter.h"
#include "Utils.h"
#include <algorithm>
#include <map>

ParticleFilter::ParticleFilter(std::shared_ptr<BuildingMap> buildingMap)
{

	o_buildingMap = buildingMap;
	const std::vector<std::shared_ptr<FloorMap>>& floors = o_buildingMap->GetFloors();
	for (int i = 0; i < floors.size(); ++i)
	{
		o_gmaps.push_back(floors[i]->Map());
	}
	o_gmap = o_gmaps[0];
	o_floorMap = floors[0];

	o_floorStats = std::vector<SetStatistics>(floors.size());
	o_floorWeights = std::vector<double>(floors.size(), 1.0 / floors.size());
}

void ParticleFilter::InitByRoomType(std::vector<Particle>& particles, int n_particles, const std::vector<float>& roomProbabilities)
{
	particles = std::vector<Particle>(n_particles);

	Eigen::Vector2f tl = o_gmap->Map2World(o_gmap->TopLeft());
	Eigen::Vector2f br = o_gmap->Map2World(o_gmap->BottomRight());

	std::vector<float> accProb = std::vector<float>(roomProbabilities.size());
	accProb[0] = roomProbabilities[0];
	for(int t = 1; t < 4; ++t)
	{
		accProb[t] = roomProbabilities[t] + accProb[t - 1]; 
	}

	int i = 0;
	while(i < n_particles)
	{
			int t = 3;
			
			float prob = drand48();
			if (prob < accProb[0]) t = 0;
			else if (prob < accProb[1]) t = 1;
			else if (prob < accProb[2]) t = 2;

			float x = drand48() * (br(0) - tl(0)) + tl(0);
			float y = drand48() * (br(1) - tl(1)) + tl(1);
			if(!o_gmap->IsValid(Eigen::Vector3f(x, y, 0))) continue;
			int roomID = o_floorMap->GetRoomID(Eigen::Vector3f(x, y, 0));
			int type = o_floorMap->GetRoom(roomID).Purpose();
			if (type != t) continue;

			float theta = drand48() * 2 * M_PI - M_PI;
			Particle p(Eigen::Vector3f(x, y, theta), 1.0 / n_particles);
			particles[i] = p;
			++i;
	}


	o_particles = particles;
}


void ParticleFilter::InitUniform(std::vector<Particle>& particles, int n_particles)
{
	particles = std::vector<Particle>(n_particles);

	int numFloor = o_gmaps.size();
	int numPerFloor = n_particles / numFloor;

	for(int m = 0; m < numFloor; ++m)
	{
		Eigen::Vector2f tl = o_gmaps[m]->Map2World(o_gmaps[m]->TopLeft());
		Eigen::Vector2f br = o_gmaps[m]->Map2World(o_gmaps[m]->BottomRight());

		int i = 0;
		while(i < numPerFloor)
		{
				float x = drand48() * (br(0) - tl(0)) + tl(0);
				float y = drand48() * (br(1) - tl(1)) + tl(1);
				if(!o_gmaps[m]->IsValid(Eigen::Vector3f(x, y, 0))) continue;

				float theta = drand48() * 2 * M_PI - M_PI;
				Particle p(Eigen::Vector3f(x, y, theta), 1.0 / n_particles, m);
				//particles[m * numPerFloor + i] = p;
				particles[i] = p;
				++i;
		}
	}

	o_particles = particles;
}


void ParticleFilter::InitGaussian(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector4f>& initGuess, const std::vector<Eigen::Matrix3d>& covariances)
{
	int totNum = n_particles * initGuess.size();
	particles = std::vector<Particle>(totNum);

	for(long unsigned int i = 0; i < initGuess.size(); ++i)
	{
		Eigen::Vector3f initG = initGuess[i].head(3);
		Eigen::Matrix3d cov = covariances[i];
		unsigned int floorID = (unsigned int)(initGuess[i][3]);

		float dx = fabs(SampleGuassian(cov(0, 0)));
		float dy = fabs(SampleGuassian(cov(1, 1)));
		float dt = fabs(SampleGuassian(cov(2, 2)));

		if (cov(2, 2) < 0.0) dt = M_PI;
		//if (cov(2, 2) > 1.0) dt = M_PI;

		Eigen::Vector3f delta(dx, dy, dt);

		Eigen::Vector3f tl = initG + delta;
		Eigen::Vector3f br = initG - delta;

		int n = 0;
		while(n < n_particles)
		{
				float x = drand48() * (br(0) - tl(0)) + tl(0);
				float y = drand48() * (br(1) - tl(1)) + tl(1);
				if(!o_gmaps[floorID]->IsValid(Eigen::Vector3f(x, y, 0))) continue;

				float theta = drand48() * (br(2) - tl(2)) + tl(2);
				Particle p(Eigen::Vector3f(x, y, theta), 1.0 / n_particles, floorID);
				particles[n + n_particles * i] = p;
				++n;
		}
	}

	o_particles = particles;
}


void ParticleFilter::AddBoundingBox(std::vector<Particle>& particles, int n_particles,  const std::vector<Eigen::Vector2f>& tls, const std::vector<Eigen::Vector2f>& brs, const std::vector<float>& yaws)
{
	int totNum = n_particles * tls.size();
	std::vector<Particle> new_particles(totNum);

	for(long unsigned int i = 0; i < tls.size(); ++i)
	{
		Eigen::Vector2f tl = o_gmap->Map2World(tls[i]);
		Eigen::Vector2f br = o_gmap->Map2World(brs[i]);
		float dx = 0.2 * drand48();
		float dy = 0.2 * drand48();

		tl += Eigen::Vector2f(-dx, dy);
		br += Eigen::Vector2f(dx, -dy);

		float yaw = yaws[i];

		int n = 0;
		while(n < n_particles)
		{
				float x = drand48() * (br(0) - tl(0)) + tl(0);
				float y = drand48() * (br(1) - tl(1)) + tl(1);
				if(!o_gmap->IsValid(Eigen::Vector3f(x, y, 0))) continue;

				float theta = 0.5 * (drand48() * 2 * M_PI - M_PI);
				theta += yaw;
				Particle p(Eigen::Vector3f(x, y, theta), 1.0 / n_particles);
				new_particles[n + n_particles * i] = p;
				++n;
		}
	}
	
	//particles.reserve(particles.size() + distance(new_particles.begin(),new_particles.end()));
	particles.insert(particles.end(),new_particles.begin(),new_particles.end());

	o_particles = particles;
}


void ParticleFilter::RemoveWeakest(std::vector<Particle>& particles, int n_particles)
{
	auto lambda = [](const Particle & a, const Particle & b) {return a.weight > b.weight; };
	std::sort(particles.begin(), particles.end(), lambda);

	int numKeep = particles.size() - n_particles;
	particles.erase(particles.begin() + numKeep, particles.end());

	o_particles = particles;
}


SetStatistics ParticleFilter::ComputeStatistics(const std::vector<Particle>& particles)
{

	std::vector<double>::iterator result = std::max_element(o_floorWeights.begin(), o_floorWeights.end());
	int floorID = std::distance(o_floorWeights.begin(), result);

	std::vector<Particle> floorParticles = GetFloorParticles(particles, floorID);
	//std::cout << "floorID: " << floorID << " particles " <<floorParticles.size() << std::endl;

	o_stats = SetStatistics::ComputeParticleSetStatistics(floorParticles);
	o_stats.FloorID(floorID);

	return o_stats;
}

std::vector<Particle> ParticleFilter::GetFloorParticles(const std::vector<Particle>& particles, unsigned int floorID)
{
	std::vector<Particle> floorParticles;
	for(int i = 0; i < particles.size(); ++i)
	{
		if (floorID == particles[i].floorID)
		{
			floorParticles.push_back(particles[i]);
		}
	}

	return floorParticles;
}


void ParticleFilter::NormalizeWeights(std::vector<Particle>& particles)
{
	// auto lambda = [&](double total, Particle p){return total + p.weight; };
	// double sumWeights = std::accumulate(o_particles.begin(), o_particles.end(), 0.0, lambda);
	// std::for_each(o_particles.begin(), o_particles.end(), [sumWeights](Particle &p){ p.weight /= sumWeights; });

	for(int f = 0; f < o_floorWeights.size(); ++f)
	{
		//std::cout << o_floorWeights[f] << std::endl;
		o_floorWeights[f] = 0.0;
	}

	double w = 0;
	for(int i = 0; i < particles.size(); ++i)
	{
		w += particles[i].weight;
	}

	for(int i = 0; i < particles.size(); ++i)
	{
		particles[i].weight = particles[i].weight / w;
		o_floorWeights[particles[i].floorID] += particles[i].weight;
	}
}
