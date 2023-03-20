/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: BuildingMap.cpp                                                       #
# ##############################################################################
**/


#include "BuildingMap.h"
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

BuildingMap::BuildingMap(std::string jsonPath)
{
    using json = nlohmann::json;

    std::string folderPath = boost::filesystem::path(jsonPath).parent_path().string() + "/";

    std::ifstream file(jsonPath);
    json config;
    file >> config;

    std::vector<int> floorIDs = config["floors"];
    for(int i = 0; i < floorIDs.size(); ++i)
    {
    	//std::cout << folderPath + std::to_string(i) + "/floor.config" << std::endl;
    	FloorMap floormap = FloorMap(folderPath + std::to_string(i) + "/floor.config");
    	std::shared_ptr<FloorMap> fp = std::make_shared<FloorMap>(floormap);
    	o_floors.push_back(fp);
    }
    

}