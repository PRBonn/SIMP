/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLFactory.cpp          				                               #
# ##############################################################################
**/


#include <fstream> 

#include "NMCLFactory.h"
#include "Resampling.h"
#include "SetStatistics.h"
#include "GMap.h"
#include "MixedFSR.h"
#include "FloorMap.h"
#include "BuildingMap.h"

#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include "SemanticOverlap.h"

using json = nlohmann::json;

std::shared_ptr<ReNMCL> NMCLFactory::Create(const std::string& configPath)
{
	std::ifstream file(configPath);
	json config;
	file >> config;
	std::string folderPath = boost::filesystem::path(configPath).parent_path().string() + "/";

	std::string motionModel = config["motionModel"];
	bool tracking = config["tracking"]["mode"];
	bool omni = config["omni"]["mode"];


	std::shared_ptr<MixedFSR> mm;
	std::shared_ptr<Resampling> rs;
	std::shared_ptr<BuildingMap> bm;
	std::shared_ptr<FloorMap> fp;
	std::shared_ptr<ReNMCL> renmcl;
	std::shared_ptr<SemanticOverlap> semantic3DModel;

	int numParticles = config["numParticles"];


	std::string jsonPath = folderPath + std::string(config["buildingMapPath"]);
	bm = std::make_shared<BuildingMap>(BuildingMap(jsonPath));
	const std::vector<std::shared_ptr<FloorMap>>& floors = bm->GetFloors();

	if(omni)
	{
		std::vector<std::string> classes = config["omni"]["classes"];
		std::vector<float> confidences = config["omni"]["confidence"];
		std::string globalVariancevPath = folderPath + "/0/global_variance.json";
		semantic3DModel = std::make_shared<SemanticOverlap>(SemanticOverlap(bm, classes, confidences, globalVariancevPath));
	}


	if(motionModel == "MixedFSR")
	{
		mm = std::make_shared<MixedFSR>(MixedFSR());
	}
	

	float th = config["resampling"]["lowVarianceTH"];
	rs = std::make_shared<Resampling>(Resampling());
	rs->SetTH(th);
	
	if(tracking)
	{
		//TODO implement these
		std::cout << "tracking" << std::endl;
		std::vector<Eigen::Matrix3d> covariances;
		std::vector<Eigen::Vector4f> initGuesses;

		std::vector<float> x = config["tracking"]["x"];
		std::vector<float> y = config["tracking"]["y"];
		std::vector<float> yaw = config["tracking"]["yaw"];
		std::vector<int> floorID = config["tracking"]["floorID"];
		std::vector<float> covx = config["tracking"]["cov_x"];
		std::vector<float> covy = config["tracking"]["cov_y"];
		std::vector<float> covyaw = config["tracking"]["cov_yaw"];

		for (int i = 0; i < x.size(); ++i)
		{
			Eigen::Matrix3d cov;
			cov << covx[i], 0, 0, 0, covy[i], 0, 0, 0, covyaw[i]; 
			covariances.push_back(cov);
			initGuesses.push_back(Eigen::Vector4f(x[i], y[i], yaw[i], floorID[i]));
		}
		renmcl = std::make_shared<ReNMCL>(ReNMCL(bm, rs, mm, semantic3DModel, numParticles, initGuesses, covariances));
	}
	else
	{
		renmcl = std::make_shared<ReNMCL>(ReNMCL(bm, rs, mm, semantic3DModel, numParticles));
	}
	
	renmcl->SetPredictStrategy(ReNMCL::Strategy::UNIFORM);
	


	std::cout << "NMCLFactory::Created Successfully!" << std::endl;

	return renmcl;
}


void NMCLFactory::Dump(const std::string& configPath)
{
	json config;
	config["motionModel"] = "MixedFSR";
	config["resampling"]["lowVarianceTH"] = 0.5;
	config["tracking"]["mode"] =  false;
	config["omni"]["mode"] =  true;
	config["predictStrategy"] = "Uniform";
	config["buildingMapPath"] = "building.config";
	config["numParticles"] = 10000;


	std::ofstream file(configPath);
	file << std::setw(4) << config << std::endl;
}