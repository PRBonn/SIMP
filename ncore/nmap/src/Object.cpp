/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: Object.cpp                                                            #
# ##############################################################################
**/

#include "Object.h"
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>

Object::Object(int semLabel, Eigen::Vector3f pose, std::string modelPath)
{
	o_id = Object::generateID();
	o_semLabel = semLabel;
	o_pose = pose;
	o_modelPath = modelPath;
}

Object::Object()
{
	o_id = Object::generateID();
	o_pose = Eigen::Vector3f(0, 0, 0);
	o_semLabel = 0;
	o_modelPath = "";
}

int Object::generateID()
{
	//srand(); 
	int id = (rand()%10000)+1; 

	return id;
}

void Object::loadModel()
{
	
}


Object::Object(nlohmann::json config)
{
	//o_id = Object::generateID();
	o_id = config["id"];
 	o_semLabel = config["semLabel"];
	std::vector<float> pos = config["position"];
	o_posistion = Eigen::Vector4f(pos[0], pos[1], pos[2], pos[3]);
}


