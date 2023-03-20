/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: Room.cpp                                                              #
# ##############################################################################
**/

#include "Room.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

Room::Room(std::string name, int id, int purpose)
{
	o_name = name;
	o_purpose = purpose;
	o_id = id;
}


Room::Room(nlohmann::json config)
{
	o_name = config["name"];
    o_purpose = config["purpose"];
    o_id = config["id"];

    auto objects = config["objects"];

    for(auto cfg : objects)
    {
    	Object obj(cfg);
    	o_objects.push_back(obj);
    }
}

Object& Room::GetObject(int id)
{
	auto it = find_if(o_objects.begin(), o_objects.end(), [&id](const Object& obj) {return obj.ID() == id;});
	auto index = std::distance(o_objects.begin(), it);
	return o_objects[index];
}


int Room::AddObject(Object& obj)
{
	o_objects.push_back(obj);
	return 0;
}

int Room::RemoveObject(int id)
{
	auto it = find_if(o_objects.begin(), o_objects.end(), [&id](const Object& obj) {return obj.ID() == id;});
	if (it != o_objects.end())
	{
	  auto index = std::distance(o_objects.begin(), it);
	  o_objects.erase (o_objects.begin() + index);

	  return 0;
	}

	return -1;
}
