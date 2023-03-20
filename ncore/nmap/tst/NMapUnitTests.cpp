/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: MSensorsUnitTests.cpp                                                            #
# ##############################################################################
**/

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <string>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "GMap.h"
#include "Utils.h"
#include "Object.h"
#include <fstream>
#include "Room.h"
#include "FloorMap.h"
#include "BuildingMap.h"
#include <nlohmann/json.hpp>


std::string dataPath = PROJECT_TEST_DATA_DIR + std::string("/8/");
std::string dataPathFloor = PROJECT_TEST_DATA_DIR + std::string("/floor/");
std::string configPath = PROJECT_TEST_DATA_DIR + std::string("/config/");
std::string testPath = PROJECT_TEST_DATA_DIR + std::string("/test/floor/");


TEST(TestGMap, test1) {
    
    Eigen::Vector3f origin = Eigen::Vector3f(-12.200000, -17.000000, 0.000000);
	float resolution = 0.05;
	cv::Mat gridMap = cv::imread(dataPath + "YouBotMap.pgm");
	GMap gmap = GMap(gridMap, origin, resolution);

	Eigen::Vector2f o = gmap.World2Map(Eigen::Vector2f(0, 0));
	ASSERT_EQ(o, Eigen::Vector2f(244, 332));
}

TEST(TestGMap, test2) {

	Eigen::Vector3f origin = Eigen::Vector3f(-12.200000, -17.000000, 0.000000);
	float resolution = 0.05;
	cv::Mat gridMap = cv::imread(dataPath + "YouBotMap.pgm");
	GMap gmap = GMap(gridMap, origin, resolution);

	Eigen::Vector2f p = gmap.Map2World(Eigen::Vector2f(244, 332));
	ASSERT_EQ(p, Eigen::Vector2f(0, 0));
}


TEST(TestGMap, test3) {

	std::string mapFolder = dataPath;
	GMap gmap = GMap(mapFolder);

	Eigen::Vector2f p = gmap.Map2World(Eigen::Vector2f(244, 332));
	ASSERT_EQ(p, Eigen::Vector2f(0, 0));
}

TEST(TestGMap, test4) {

   std::string mapFolder = dataPath;
    GMap gmap = GMap(mapFolder);

    Eigen::Vector2f p = gmap.Map2World(Eigen::Vector2f(244, 332));
    Eigen::Vector2f pm = gmap.World2Map(p);

    ASSERT_EQ(pm(0), 244);
    ASSERT_EQ(pm(1), 332);
}

TEST(TestGMap, test5)
{

    std::string mapFolder = dataPath;
    GMap gmap = GMap(mapFolder);

    Eigen::Vector2f p = gmap.Map2World(Eigen::Vector2f(244, 332));
    bool valid = gmap.IsValid(Eigen::Vector3f(p(0), p(1), 0));

    ASSERT_EQ(true, valid);
}

TEST(TestGMap, test6)
{

    std::string mapFolder = dataPath;
    GMap gmap = GMap(mapFolder);

    Eigen::Vector2f p = gmap.Map2World(Eigen::Vector2f(0, 0));
    bool valid = gmap.IsValid(Eigen::Vector3f(p(0), p(1), 0));

    ASSERT_EQ(false, valid);
}


TEST(TestObject, test1) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);
    int id = ob.ID();


    std::string filename = "object.xml";
    std::ofstream ofs(filename.c_str());
    boost::archive::text_oarchive oa(ofs);
    oa << ob;
    ofs.close();


    Object ob2(6);
    std::ifstream ifs(filename.c_str());
    boost::archive::text_iarchive ia(ifs);
    ia >> ob2;
    ifs.close();

    int id2 = ob2.ID();
    int semLabel2 = ob2.SemLabel();
    Eigen::Vector3f pose2 = ob2.Pose();
    std::string modelPath2 = ob2.ModelPath();


    ASSERT_EQ(id, id2);
    ASSERT_EQ(semLabel, semLabel2);
    ASSERT_EQ(pose, pose2);
    ASSERT_EQ(modelPath, modelPath2);

}

TEST(TestObject, test2) {
    
    std::string jsonPath = testPath + "object.config";
   // Object obj(jsonPath);

    using json = nlohmann::json;
    std::ifstream file(jsonPath);
    json config;
    file >> config;

    Object obj(config);

    Eigen::Vector4f pos = obj.Position();

    ASSERT_EQ(obj.SemLabel(), 0);
    ASSERT_EQ(pos(0), 1);
    ASSERT_EQ(pos(1), 5);

}


TEST(TestRoom, test1) {
    
    std::string name = "room";
    int roomID = 9;
    Room r(name, roomID);


    std::string filename = "room.xml";
    std::ofstream ofs(filename.c_str());
    boost::archive::text_oarchive oa(ofs);
    oa << r;
    ofs.close();

    Room r2("room2", 11);
    std::ifstream ifs(filename.c_str());
    boost::archive::text_iarchive ia(ifs);
    ia >> r2;
    ifs.close();

    ASSERT_EQ(roomID, r2.ID());
    ASSERT_EQ(name, r2.Name());

}


TEST(TestRoom, test2) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);
    int id = ob.ID();

    int semLabel2 = 7;
    Eigen::Vector3f pose2(1.3, -10.0, -2.7);
    std::string modelPath2 = "/nicky/poo2.pcl";
    Object ob2(semLabel2, pose2, modelPath2);
  
    std::string name = "room";
    int roomID = 9;
    Room r(name, roomID);
    r.AddObject(ob);
    r.AddObject(ob2);


    std::string filename = "room.xml";
    std::ofstream ofs(filename.c_str());
    boost::archive::text_oarchive oa(ofs);
    oa << r;
    ofs.close();

    Room r2("room2", 11);
    std::ifstream ifs(filename.c_str());
    boost::archive::text_iarchive ia(ifs);
    ia >> r2;
    ifs.close();

    std::vector<Object> objects = r2.Objects();

    ASSERT_EQ(roomID, r2.ID());
    ASSERT_EQ(name, r2.Name());

    ASSERT_EQ(id, objects[0].ID());
    ASSERT_EQ(semLabel2, objects[1].SemLabel());

}

TEST(TestRoom, test3) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);

    int semLabel2 = 7;
    Eigen::Vector3f pose2(1.3, -10.0, -2.7);
    std::string modelPath2 = "/nicky/poo2.pcl";
    Object ob2(semLabel2, pose2, modelPath2);

    std::string name = "room";
    int roomID = 9;
    Room r(name, roomID);
    r.AddObject(ob);
    r.AddObject(ob2);

    int size = r.Objects().size();
    ASSERT_EQ(2, size);
}

TEST(TestRoom, test4) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);
    int id = ob.ID();

    int semLabel2 = 7;
    Eigen::Vector3f pose2(1.3, -10.0, -2.7);
    std::string modelPath2 = "/nicky/poo2.pcl";
    Object ob2(semLabel2, pose2, modelPath2);
    int id2 = ob2.ID();


    std::string name = "room";
    int roomID = 9;
    Room r(name, roomID);
    r.AddObject(ob);
    r.AddObject(ob2);


    r.RemoveObject(id);
    int size = r.Objects().size();

    ASSERT_EQ(1, size);
    ASSERT_EQ(id2, r.Objects()[0].ID());
}

TEST(TestRoom, test5) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);

    int semLabel2 = 7;
    Eigen::Vector3f pose2(1.3, -10.0, -2.7);
    std::string modelPath2 = "/nicky/poo2.pcl";
    Object ob2(semLabel2, pose2, modelPath2);
 
    std::string name = "room";
    int roomID = 9;
    Room r(name, roomID);
    r.AddObject(ob);
    r.AddObject(ob2);

    int badID = 0;


    int res = r.RemoveObject(badID);
    int size = r.Objects().size();

    ASSERT_EQ(2, size);
    ASSERT_EQ(-1, res);
}

TEST(TestRoom, test6) {
    
    std::string jsonPath = testPath + "room.config";

    using json = nlohmann::json;
    std::ifstream file(jsonPath);
    json config;
    file >> config;

    Room room(config);


    ASSERT_EQ(room.Name(), "Room 1");
    ASSERT_EQ(room.ID(), 5);

    std::vector<Object> objects = room.Objects();
    ASSERT_EQ(objects[1].SemLabel(), 1);

}

/*
TEST(TestFloorMap, test1) {
    
    int semLabel = 10;
    Eigen::Vector3f pose(1.5, 0.0, 2.7);
    std::string modelPath = "/nicky/poo.pcl";
    Object ob(semLabel, pose, modelPath);
    int id = ob.ID();

    int semLabel2 = 7;
    Eigen::Vector3f pose2(1.3, -10.0, -2.7);
    std::string modelPath2 = "/nicky/poo2.pcl";
    Object ob2(semLabel2, pose2, modelPath2);

    GMap gmap = GMap(dataPathFloor + "Test/", "FMap.yaml");

    cv::Mat roomSeg = cv::imread(dataPathFloor + "Test/FMapRoomSeg.png");
    FloorMap fp = FloorMap(std::make_shared<GMap>(gmap), roomSeg);

    fp.GetRoom(1).AddObject(ob);
    fp.GetRoom(1).Name("room");

    fp.GetRoom(2).AddObject(ob2);
    fp.GetRoom(2).Name("room2");

    std::string filename = dataPathFloor + "Test/" + "floor.xml";
    std::ofstream ofs(filename.c_str());
    boost::archive::text_oarchive oa(ofs);
    oa << fp;
    ofs.close();

    FloorMap fp2 = FloorMap(std::make_shared<GMap>(gmap), roomSeg);
    std::ifstream ifs(filename.c_str());
    boost::archive::text_iarchive ia(ifs);
    ia >> fp2;
    ifs.close();

    std::vector<Object> objects = fp.GetRoom(1).Objects();
    std::vector<Room> rooms = fp.GetRooms();

    ASSERT_EQ(13, rooms.size());
    ASSERT_EQ("room2", fp.GetRoom(2).Name());

    ASSERT_EQ(id, objects[0].ID());

}

TEST(TestFloorMap, test2) {
    
    GMap gmap = GMap(dataPathFloor + "Test/", "FMap.yaml");
    cv::Mat roomSeg = cv::imread(dataPathFloor + "Test/FMapRoomSeg.png");
    FloorMap fp = FloorMap(std::make_shared<GMap>(gmap), roomSeg);
    
    float x = 440;
    float y = 50;
    int id = fp.GetRoomID(x, y);

    ASSERT_EQ(1, id);

}


TEST(TestFloorMap, test3) {
    

    GMap gmap = GMap(dataPathFloor + "Test/", "FMap.yaml");
    cv::Mat roomSeg = cv::imread(dataPathFloor + "Test/FMapRoomSeg.png");
    FloorMap fp = FloorMap(std::make_shared<GMap>(gmap), roomSeg);
    
    fp.GetRoom(1).Name("room");
    fp.GetRoom(2).Name("room2");

    ASSERT_EQ("room2", fp.GetRoom(2).Name());

}

TEST(TestFloorMap, test4) {
    
    GMap gmap = GMap(dataPathFloor + "Test/", "FMap.yaml");
    cv::Mat roomSeg = cv::imread(dataPathFloor + "Test/FMapRoomSeg.png");
    FloorMap fp = FloorMap(std::make_shared<GMap>(gmap), roomSeg);
    fp.findNeighbours();

    std::vector<std::vector<int>> neighbours = fp.Neighbors();
    //std::cout << fp.Neighbors().size() << std::endl;
    // for (int i = 0; i < neighbours.size(); ++i)
    // {
    //     std::cout << "Room " << i + 1 << " neighbours:" << std::endl;
    //     for(int j = 0; j < neighbours[i].size(); ++j)
    //     {
    //         std::cout << neighbours[i][j] + 1 << ", " << std::endl;
    //     }
    //     std::cout <<  std::endl;
    // }

    ASSERT_EQ(13, fp.Neighbors().size());
    ASSERT_EQ(2, fp.Neighbors()[0][0] + 1);

}


TEST(TestFloorMap, test5) {
    
    std::string jsonPath = dataPathFloor + "Test/" + "floor.config";
    FloorMap fp = FloorMap(jsonPath);
   
    fp.findNeighbours();
    std::vector<std::vector<int>> neighbours = fp.Neighbors();

    ASSERT_EQ(13, fp.Neighbors().size());
    ASSERT_EQ(2, fp.Neighbors()[0][0] + 1);

}

TEST(TestFloorMap, test6) {
    
    std::string jsonPath = "/home/nickybones/Code/OmniNMCL/ncore/data/test/floor.config";

    using json = nlohmann::json;
    std::ifstream file(jsonPath);
    json config;
    file >> config;

    FloorMap floor(config, "/home/nickybones/Code/OmniNMCL/ncore/data/test/");

    std::vector<std::string> classes = floor.Classes();
    std::vector<Room> rooms =  floor.GetRooms();


    ASSERT_EQ(classes[0], "sink");
    ASSERT_EQ(rooms[0].Name(), "Room 1");

    std::vector<Object> objects = rooms[1].Objects();
    ASSERT_EQ(objects[1].SemLabel(), 4);

}*/

TEST(TestFloorMap, test1) {
    
    std::string jsonPath = testPath + "floor.config";

    using json = nlohmann::json;
    std::ifstream file(jsonPath);
    json config;
    file >> config;

    FloorMap floor(config, testPath);

    std::vector<std::string> classes = floor.Classes();
    std::vector<Room> rooms =  floor.GetRooms();


    ASSERT_EQ(classes[0], "sink");
    ASSERT_EQ(rooms[11].Name(), "Room 10");

    std::vector<Object> objects = rooms[1].Objects();
    ASSERT_EQ(objects[1].SemLabel(), 1);

}

TEST(TestFloorMap, test2) {
    
    std::string jsonPath = testPath + "floor.config";

    FloorMap floor(jsonPath);

    std::vector<std::string> classes = floor.Classes();
    std::vector<Room> rooms =  floor.GetRooms();


    ASSERT_EQ(classes[0], "sink");
    ASSERT_EQ(rooms[11].Name(), "Room 10");
    std::vector<Object> objects = rooms[1].Objects();
    ASSERT_EQ(objects[1].SemLabel(), 1);

}


TEST(TestBuildingMap, test1) {
    
    std::string jsonPath = testPath + std::string("building.config");

    BuildingMap bm(jsonPath);

    std::vector<std::shared_ptr<FloorMap>> floors = bm.GetFloors();

    std::vector<std::string> classes = floors[0]->Classes();
    std::vector<Room> rooms =  floors[0]->GetRooms();


    ASSERT_EQ(classes[0], "sink");
    ASSERT_EQ(rooms[11].Name(), "Room 10");

    std::vector<Object> objects = rooms[1].Objects();
    ASSERT_EQ(objects[1].SemLabel(), 1);

}


TEST(TestBuildingMap, test2) {
    
    std::string jsonPath = std::string("/home/nickybones/Code/OmniNMCL/ncore/data/floor/ETHNano/building.config");

    BuildingMap bm(jsonPath);

    std::vector<std::shared_ptr<FloorMap>> floors = bm.GetFloors();

    std::vector<cv::Mat> classMaps = floors[0]->CreateSemMaps();

    std::vector<std::string> classNames = {"sink", "door", "oven", "whiteboard", "table", "cardboard", "plant", "drawers", "sofa", "storage", "chair", "extinguisher", "people", "desk"};


    for(int i = 0; i < classMaps.size(); ++i)
    {
       cv::imwrite("/home/nickybones/Code/OmniNMCL/ncore/data/floor/ABB/0/SemMaps/" + classNames[i] + ".png", classMaps[i]);
    }

    // ASSERT_EQ(classes[0], "sink");
    // ASSERT_EQ(rooms[11].Name(), "Room 10");

    // std::vector<Object> objects = rooms[1].Objects();
    // ASSERT_EQ(objects[1].SemLabel(), 1);

}










int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}



