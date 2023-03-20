/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: FloorMap.cpp                                                          #
# ##############################################################################
**/

#include "FloorMap.h"
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <boost/filesystem.hpp>



FloorMap::FloorMap(nlohmann::json config, std::string folderPath)
{
    o_name = config["name"];
    o_folderPath = folderPath;

    std::string segPath = config["roomSeg"];
    cv::Mat roomSeg = cv::imread(folderPath + segPath);
    cv::Mat split[3];
    cv::split(roomSeg, split);
    o_roomSeg = split[2];

    std::vector<std::string> classes = config["semantic"]["classes"];
    std::vector<std::string> categories = config["semantic"]["categories"];
    o_classes = classes;
    o_categories = categories;

    //std::vector<int> roomIDs = extractRoomIDs();

    // Room 0 is background
    o_rooms.push_back(Room("NotValid", -1, -1));
    auto rooms = config["rooms"];

    for(auto cfg : rooms)
    {
        Room room(cfg);
        //std::cout << room.ID() << std::endl;
        o_rooms.push_back(room);
    }

    auto maps = config["map"];
    for(auto cfg : maps)
    {
        if(cfg["type"] == "GMap")
        {
            std::shared_ptr<GMap> map;
            float resolution = cfg["resolution"];
            std::vector<float> origin = cfg["origin"];
            std::string imgPath = cfg["image"];
            cv::Mat img = cv::imread(folderPath + imgPath);
            int augment = cfg["augment"];
            if (augment)
            {
                std::vector<int> augmentedClasses = {9,8};
                cv::Mat augmentedGmap = augmentGMap(img, augmentedClasses);
                cv::imwrite("augmentedGmap.png", augmentedGmap);
                map = std::make_shared<GMap>(GMap(augmentedGmap, Eigen::Vector3f(origin[0], origin[1], origin[2]), resolution));
            }
            else
            {
                map = std::make_shared<GMap>(GMap(img, Eigen::Vector3f(origin[0], origin[1], origin[2]), resolution));
            }
            o_maps.push_back(map);
        }
    }
    o_map = o_maps[0];

}

FloorMap::FloorMap(std::string jsonPath)
{
    using json = nlohmann::json;

    o_folderPath = boost::filesystem::path(jsonPath).parent_path().string() + "/";

    std::ifstream file(jsonPath);
    json config;
    file >> config;

    o_name = config["name"];

    std::string segPath = config["roomSeg"];
    cv::Mat roomSeg = cv::imread(o_folderPath + segPath);
    cv::Mat split[3];
    cv::split(roomSeg, split);
    o_roomSeg = split[2];

    std::vector<std::string> classes = config["semantic"]["classes"];
    std::vector<std::string> categories = config["semantic"]["categories"];
    o_classes = classes;
    o_categories = categories;

    //std::vector<int> roomIDs = extractRoomIDs();

    // Room 0 is background
    o_rooms.push_back(Room("NotValid", -1, -1));
    auto rooms = config["rooms"];

    for(auto cfg : rooms)
    {
        Room room(cfg);
        //std::cout << room.ID() << std::endl;
        o_rooms.push_back(room);
    }

    auto maps = config["map"];
    for(auto cfg : maps)
    {
        //std::cout << cfg << std::endl;
        if(cfg["type"] == "GMap")
        {
            std::shared_ptr<GMap> map;
           // std::cout << cfg["image"] <<std::endl;
            float resolution = cfg["resolution"];
            std::vector<float> origin = cfg["origin"];
            std::string imgPath = cfg["image"];
            cv::Mat img = cv::imread(o_folderPath + imgPath);
            int augment = cfg["augment"];
            if (augment)
            {
                std::vector<int> augmentedClasses = {9, 8};
                cv::Mat augmentedGmap = augmentGMap(img, augmentedClasses);
                cv::imwrite("augmentedGmap.png", augmentedGmap);
                map = std::make_shared<GMap>(GMap(augmentedGmap, Eigen::Vector3f(origin[0], origin[1], origin[2]), resolution));
            }
            else
            {
                map = std::make_shared<GMap>(GMap(img, Eigen::Vector3f(origin[0], origin[1], origin[2]), resolution));
            }
            o_maps.push_back(map);
        }
    }
    o_map = o_maps[0];

}

cv::Mat FloorMap::augmentGMap(const cv::Mat& img, const std::vector<int>& augmentedClasses)
{
    cv::Mat augmentedGmap = img.clone();
    cv::Mat edges = cv::Mat::zeros(augmentedGmap.size(), CV_8U);
    int padding = 3;
    int lowThreshold = 50;
    const int max_lowThreshold = 100;
    const int ratio = 3;
    const int kernel_size = 3;

    for(auto room : o_rooms)
    {
        const std::vector<Object>& objs = room.Objects();
        if (objs.size())
        {
            int roomID = room.ID();
            cv::Mat roomMapBin = o_roomSeg.clone();
            cv::threshold(roomMapBin, roomMapBin, roomID + 1, 255, 4);
            cv::threshold(roomMapBin, roomMapBin, roomID, 255, 0);
            //cv::imwrite("Room" + std::to_string(roomID) + ".png", roomMap);

            cv::Mat augmentedRoom = img.clone();
          //  cv::Mat roomEdges;
          //  cv::Canny( roomMapBin, roomEdges, lowThreshold, lowThreshold*ratio, kernel_size );


            for(auto obj : objs)
            {
                int semID = obj.SemLabel();
                if (std::find(begin(augmentedClasses), end(augmentedClasses), semID) != std::end(augmentedClasses))
                {
                    Eigen::Vector4f pos = obj.Position();
                    cv::Point pt1(pos(0) -padding , pos(1)- padding);
                    cv::Point pt2(pos(2)+padding, pos(3)+padding);
                    cv::rectangle(augmentedRoom, pt1, pt2, cv::Scalar(205, 205, 205), -1);
                    //cv::rectangle(augmentedGmap, pt1, pt2, cv::Scalar(0, 0,0), 1);
                }
            }
         augmentedRoom.copyTo(augmentedGmap, roomMapBin);
        // roomEdges.copyTo(edges, roomMapBin);
        }
    }
    cv::cvtColor(augmentedGmap, augmentedGmap, cv::COLOR_BGR2GRAY);
    cv::threshold(augmentedGmap, augmentedGmap, 254, 255, 0);
    cv::Mat unknown = 205 * cv::Mat::ones(augmentedGmap.size(), CV_8U);
    cv::Mat occupied = cv::Mat::zeros(augmentedGmap.size(), CV_8U);

    cv::Canny( augmentedGmap, edges, lowThreshold, lowThreshold*ratio, kernel_size );

    augmentedGmap = augmentedGmap + unknown;
    occupied.copyTo(augmentedGmap, edges);

    cv::imwrite("augmentedGmap.png", augmentedGmap);
    cv::imwrite("edges.png", edges);
    return augmentedGmap;
}

std::vector<cv::Mat> FloorMap::CreateSemMaps()
{
    int h = o_roomSeg.rows;
    int w = o_roomSeg.cols;
    std::vector<cv::Mat> semMaps = std::vector<cv::Mat>(o_classes.size());

    for(int c = 0; c < o_classes.size(); ++c)
    {    
        semMaps[c] = cv::Mat::zeros(o_roomSeg.size(), CV_8UC1);
    }

    for (int r = 0; r <  o_rooms.size(); ++r)
    {
        Room room = o_rooms[r];
        const std::vector<Object> objs = room.Objects();
        for (int o = 0; o <  objs.size(); ++o)
        {
            int semLabel = objs[o].SemLabel();
            Eigen::Vector4f pos = objs[o].Position();
            cv::Rect rect(pos(0), pos(1), pos(2) - pos(0), pos(3) - pos(1));
            cv::rectangle(semMaps[semLabel], rect, 255, -1);
        }
    }

    // for(int c = 0; c < o_classes.size(); ++c)
    // {    
    //     cv::imwrite(o_folderPath + "SemMaps/" + o_classes[c] + ".png", semMaps[c]);
    // }

    return semMaps;
}


int FloorMap::GetRoomID(float x, float y)
{
    uchar val = o_roomSeg.at<uchar>(y, x);
    return val;
}

int FloorMap::GetRoomID(Eigen::Vector3f pose)
{
    Eigen::Vector2f uv = o_map->World2Map(Eigen::Vector2f(pose(0), pose(1)));
    uchar val = o_roomSeg.at<uchar>(uv(1), uv(0));
    return val;
}


void FloorMap::extractRoomSegmentation()
{
	std::vector<int> roomIDs = extractRoomIDs();

	for (long unsigned int i = 0; i < roomIDs.size(); ++i)
	{
		int id = roomIDs[i];
		o_rooms.push_back(Room(std::to_string(id), id));
	}
}



std::vector<int> FloorMap::extractRoomIDs()
{
    cv::Mat flat = o_roomSeg.reshape(1, o_roomSeg.total() * o_roomSeg.channels());
    std::vector<uchar> vec = o_roomSeg.isContinuous() ? flat : flat.clone();
    std::set<uchar> s( vec.begin(), vec.end() );
    vec.assign( s.begin(), s.end() );
    std::sort(vec.begin(), vec.end());

    std::vector<int> roomIDs;
    //roomIDs.push_back(0);

    for(long unsigned int i = 1; i < vec.size(); ++i)
    {
       
        if (vec[i] == vec[i - 1] + 1)
        {
            roomIDs.push_back(i);
        }
        else break;
    }

    return roomIDs;
}

void FloorMap::findNeighbours()
{
    std::vector<int> roomIDs = extractRoomIDs();
    std::vector<cv::Rect> boundRect(roomIDs.size());
    std::vector<cv::RotatedRect> rotRect;

    cv::Mat orig = cv::Mat::zeros( o_roomSeg.size(), CV_8UC3 );

    for(long unsigned int r = 0; r < roomIDs.size(); ++r)
    {
        int roomID = roomIDs[r];
        cv::Mat dst;
        cv::threshold( o_roomSeg, dst, roomID, 255, 4 );
        cv::threshold( dst, dst, roomID - 1, 255, 0 );

        cv::Mat threshold_output;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours( dst, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
        int bigContID = 0;
        double maxArea = 0;
        cv::RNG rng(12345);
        cv::Mat drawing = cv::Mat::zeros( dst.size(), CV_8UC3 );

        for( size_t i = 0; i < contours.size(); i++ )
        {
            double newArea = cv::contourArea(contours[i]);
            if (newArea > maxArea)
            {
                bigContID = i;
                maxArea = newArea;
            }
        }
        cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
     
        cv::RotatedRect box = cv::minAreaRect(contours[bigContID]); 
        cv::Point2f vertices[4];
        cv::Point2f center = box.center;
        box.points(vertices);

        std::vector<cv::Point> scaledVertices;
        float scale = 1.2;
        for(int w = 0; w < 4;  ++w)
        {
            cv::Point p = scale * (vertices[w] - center) + center;
            scaledVertices.push_back(p);
        }

        cv::RotatedRect scaledBox = cv::minAreaRect(scaledVertices);
        rotRect.push_back(scaledBox);

        // scaledBox.points(vertices);
        // for (int j = 0; j < 4; j++)
        // {
        //     cv::line(orig, vertices[j], vertices[(j+1)%4], color, 2);
        // }
    }

    //cv::imwrite("boxes.png", orig);
    o_neighbors = std::vector<std::vector<int>>(roomIDs.size());

    for(long unsigned int r = 0; r < roomIDs.size(); ++r)
    {
        cv::RotatedRect A = rotRect[r];
        for(long unsigned int s = 0; s < roomIDs.size(); ++s)
        {
            if (s == r) continue;
    
            cv::RotatedRect B = rotRect[s];
            std::vector<cv::Point2f> intersection;
            cv::rotatedRectangleIntersection(A, B, intersection);
            if (intersection.size())
            {
                o_neighbors[r].push_back(int(s));
            }
        }
    }
}


cv::Mat FloorMap::ColorizeRoomSeg()
{
    cv::RNG rng(12345);
    cv::Mat roomSegRGB;
    cv::cvtColor(o_roomSeg, roomSegRGB, cv::COLOR_GRAY2BGR);


    int maxElm = o_rooms.size();
    std::vector<cv::Scalar> colors;
    for(int i = 1; i <= maxElm; ++i)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        colors.push_back(color);
    }

    for(int i = 0; i < o_roomSeg.rows; i++)
    {
        for(int j = 0; j < o_roomSeg.cols; j++)
        {
            uchar val = o_roomSeg.at<uchar>(i, j);
            if ((val >= 1) && (val <= maxElm))
            {
                cv::Scalar color = colors[val - 1];
                roomSegRGB.at<cv::Vec3b>(i, j) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }

    return roomSegRGB;
}


std::vector<std::string> FloorMap::GetRoomNames()
{
    std::vector<std::string> places;
    for(int i = 0; i < o_rooms.size(); ++i)
    {
        places.push_back(o_rooms[i].Name());
    }
    return places;
}


void FloorMap::Seed(int id, const Eigen::Vector2f& seed)
{
    Eigen::Vector2f p = o_map->Map2World(seed);
    o_seeds[id] = Eigen::Vector3f(p(0), p(1), 0);
}