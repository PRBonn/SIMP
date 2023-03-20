/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: FloorMap.h                                                            #
# ##############################################################################
**/

#ifndef FLOORMAP
#define FLOORMAP
#pragma once

#include <memory>
#include <vector>

#include "GMap.h" 
#include "Room.h"
#include "Lift.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

class FloorMap
{
public:

	FloorMap(std::string jsonPath);

	FloorMap(nlohmann::json config, std::string folderPath);


	int GetRoomID(float x, float y);

	int GetRoomID(Eigen::Vector3f pose);

	std::vector<cv::Mat> CreateSemMaps();

	std::string Name() const
	{
		return o_name;
	}

	Room& GetRoom(int id)
	{
		return o_rooms[id];
	}

	std::vector<Room>& GetRooms()
	{
		return o_rooms;
	}

	int GetRoomsNum() const
	{
		return o_rooms.size();
	}

	std::vector<std::string> GetRoomNames();

	const std::vector<std::vector<int>>& Neighbors()
	{
		return o_neighbors;
	}

	// returned seeds in world coordinates
	const std::vector<Eigen::Vector3f>& Seeds() const
	{
		return o_seeds;
	}

	// returned seed in world coordinates
	const Eigen::Vector3f& Seed(int id) const
	{
		return o_seeds[id];
	}

	// input seed in world coordinates
	void Seed(int id, const Eigen::Vector3f& seed) 
	{
		o_seeds[id] = seed;
	}

	// input seed in map ccordinates
	void Seed(int id, const Eigen::Vector2f& seed);

	// input seeds in world coordinates
	void Seeds(std::vector<Eigen::Vector3f> seeds) 
	{
		o_seeds = seeds;
	}

	const std::shared_ptr<GMap>& Map() const
	{
		return o_map;
	}

	const std::vector<std::shared_ptr<GMap>>& Maps() const
	{
		return o_maps;
	}

	const std::vector<std::string>& Classes() const
	{
		return o_classes;
	}



	cv::Mat ColorizeRoomSeg();
	void findNeighbours();


private:

	friend class boost::serialization::access;
	template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & o_rooms;
        ar & o_seeds;
    }

	void extractRoomSegmentation();
	std::vector<int> extractRoomIDs();
	cv::Mat augmentGMap(const cv::Mat& img, const std::vector<int>& augmentedClasses);
	//void fingNeighbours();


	std::string o_name = "0";
	std::shared_ptr<GMap> o_map;
	std::vector<std::shared_ptr<GMap>> o_maps;
	cv::Mat o_roomSeg;
	std::vector<Room> o_rooms;
	//std::vector<Lift> o_lifts;
	std::vector<std::vector<int>> o_neighbors;
	std::vector<Eigen::Vector3f> o_seeds;
	std::vector<std::string> o_classes;
	std::vector<std::string> o_categories;
	std::string o_folderPath;


};

#endif // !FLOORMAP

