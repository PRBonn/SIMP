/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: RoomSegmentation.cpp                                                  #
# ##############################################################################
**/

// RoomSegmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include "opencv2/opencv.hpp"

#include "GMap.h"
#include "FloorMap.h"


int main()
{
    cv::Mat grid = cv::imread("/home/nickybones/Code/YouBotMCL/ncore/data/floor/2021_10_27/YouBotMap.pgm");
    cv::Mat roomSeg = cv::imread("/home/nickybones/Code/YouBotMCL/ncore/data/floor/2021_10_27/YouBotMapRoomSeg.png");

    //int id = o_roomSeg.at<uchar>(440, 300);
    //std::cout << id << std::endl;

    // cv::Mat split[3];
    // cv::split(roomSeg3, split);
    // cv::Mat roomSeg = split[2];

    GMap map = GMap(grid, Eigen::Vector3f(-17.000000, -12.200000, 0.000000), 0.05);
    FloorMap fm = FloorMap(std::make_shared<GMap>(map), roomSeg);
    cv::Mat roomSegRGB = fm.ColorizeRoomSeg();

    cv::imshow("room segmentation", roomSegRGB);
    cv::imwrite("colorized_room_seg.png", roomSegRGB);
    cv::waitKey();

}

