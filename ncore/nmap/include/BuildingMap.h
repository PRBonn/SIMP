/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: BuildingMap.h                                                         #
# ##############################################################################
**/

#ifndef BUILDINGMAP
#define BUILDINGMAP
#pragma once

#include <memory>
#include <string>
#include "FloorMap.h"

class BuildingMap
{
public:

	BuildingMap(std::string jsonPath);

	const std::vector<std::shared_ptr<FloorMap>>& GetFloors()
	{
		return o_floors;
	}


private:

	std::vector<std::shared_ptr<FloorMap>> o_floors;


};

#endif