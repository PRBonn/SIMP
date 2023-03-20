/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLSanityTests.cpp           		                           		   #
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

#include "FSR.h"
#include "Utils.h"
#include "Lidar2D.h"
#include "OptiTrack.h"
#include "GMap.h"

#include "BeamEnd.h"
#include "NMCL.h"
#include "Dataset.h"
#include "Analysis.h"
#include "MixedFSR.h"
#include "SetStatistics.h"
#include "VODataSet.h"
#include "LowVarianceResampling.h"
#include "Camera.h"

std::string dataPath = PROJECT_TEST_DATA_DIR + std::string("/8/");
std::string dataPath2 = PROJECT_TEST_DATA_DIR + std::string("/8_high_res/");
std::string configPath = PROJECT_TEST_DATA_DIR + std::string("/config/");


TEST(TestNMCL, test1)
{
	int n = 100;

	 GMap gmap = GMap(dataPath2);

	BeamEnd be = BeamEnd(std::make_shared<GMap>(gmap), 32, 15);
	FSR fsr = FSR();
	Eigen::Vector3f noise = Eigen::Vector3f(0.1, 0.1, 0.1);
	LowVarianceResampling lvr = LowVarianceResampling();


	std::string optiPath = std::string(dataPath + "opti.bin");
	std::string odomPath = std::string(dataPath + "odom.bin");
	std::string scanfPath = std::string(dataPath + "scan_front.bin");
	std::string scanrPath = std::string(dataPath + "scan_rear.bin");

	Dataset ds = Dataset();
	ds.ReadPythonBinary(odomPath, scanfPath, scanrPath, optiPath);
	const std::vector<std::vector<float>> scanFront = ds.ScanFront();
	const std::vector<std::vector<float>> scanRear = ds.ScanRear();
	const std::vector<std::vector<float>> Odoms = ds.Odom();
	const std::vector<std::vector<float>> opti = ds.OptiTrack();
	

	int dsFactor = 10;
	Lidar2D l2d_f = Lidar2D("front_laser", configPath);
	Lidar2D l2d_r = Lidar2D("rear_laser", configPath);
	std::vector<double> mask(2 * l2d_f.Heading().size(), 1.0);
	std::vector<double> scanMask = Downsample(mask, dsFactor);

	std::vector<float> commandWeights{1.0f};

	int num_frames = 300;
	int end = Odoms.size() - num_frames - 1;
	Eigen::Vector3f err = Eigen::Vector3f(0, 0, 0);
	int nExp = 50;
	int cnt = 0;

	for(int t = 0; t < nExp; ++t)
	{
		int start = int(lrand48() % end) + 1;
	
		std::vector<Eigen::Vector3f> initGuesses{Eigen::Vector3f(opti[start][0], opti[start][1], opti[start][2] )};
		Eigen::Matrix3d cov;
		cov << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5;
		std::vector<Eigen::Matrix3d> covariances{cov};

		NMCL nmcl = NMCL(std::make_shared<FSR>(fsr), std::make_shared<BeamEnd>(be), std::make_shared<LowVarianceResampling>(lvr), n, initGuesses, covariances);
		std::vector<Particle> particles = nmcl.Particles();
		//be.plotParticles(particles, std::to_string(0), true);

		std::vector<float> odom = Odoms[start - 1];
		Eigen::Vector3f prevPose = Eigen::Vector3f(odom[0], odom[1], GetYaw(odom[5], odom[6]));
		

		std::vector<Eigen::Vector3f> predictions;
		std::vector<Eigen::Vector3f> groundTruth;

		for(int i = start; i < start + num_frames; ++i)
		{

			std::vector<float> odom = Odoms[i];
			Eigen::Vector3f currPose = Eigen::Vector3f(odom[0], odom[1], GetYaw(odom[5], odom[6]));
			Eigen::Vector3f delta = currPose - prevPose;

			if(((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > 0.05) || (fabs(delta(2)) > 0.05))
			{
				Eigen::Vector3f u = fsr.Backward(prevPose, currPose);
				std::vector<Eigen::Vector3f> command{u};

				nmcl.Predict(command, commandWeights, noise);
				std::vector<Eigen::Vector3f> points_3d = MergeScans(scanFront[i], l2d_f, scanRear[i], l2d_r, dsFactor);
				nmcl.Correct(points_3d, scanMask);
				prevPose = currPose;

			
				SetStatistics stas = nmcl.Stats();
				Eigen::Matrix3d cov = stas.Cov();
				Eigen::Vector3d pred = stas.Mean();

				if(pred.array().isNaN().any())
				{
					std::cout << "Localization failed!" << std::endl;
					break;
				}
				else
				{
					predictions.push_back(Eigen::Vector3f(pred(0), pred(1), pred(2)));
					std::vector<float> GT = opti[i];
					groundTruth.push_back(Eigen::Vector3f(GT[0], GT[1], GT[2]));				}

				// std::cout << "GT: " << GT[0] << ", " << GT[1] << ", " << GT[2] << std::endl;
				// std::cout << "Pred: " << pred[0] << ", " << pred[1] << ", " << pred[2] << std::endl;
				// std::cout << "Odom: " << prevPose[0] << ", " << prevPose[1] << ", " << prevPose[2] << std::endl;
			}
			if (i % 100 == 0)
			{
				std::vector<Particle> particles = nmcl.Particles();
				//be.plotParticles(particles, std::to_string(i), false);
			}		
		}
		if (predictions.size() > 10)
		{
			err += RMSE(predictions, groundTruth);
			++cnt;
		}
	}

	err = err / cnt;

	std::cout << "Avg rmse: " << err(0)<< ", " << err(1) << ", " << err(2) << std::endl;
	ASSERT_LT(err.norm(), 0.25);	
}


TEST(TestNMCL, test2)
{
	int n = 10000;

	GMap gmap = GMap(dataPath);

	BeamEnd be = BeamEnd(std::make_shared<GMap>(gmap));
	FSR fsr = FSR();
	Eigen::Vector3f noise = Eigen::Vector3f(0.1, 0.1, 0.1);

	LowVarianceResampling lvr = LowVarianceResampling();

	std::string optiPath = std::string(dataPath + "opti.bin");
	std::string odomPath = std::string(dataPath + "odom.bin");
	std::string scanfPath = std::string(dataPath + "scan_front.bin");
	std::string scanrPath = std::string(dataPath + "scan_rear.bin");

	Dataset ds = Dataset();
	ds.ReadPythonBinary(odomPath, scanfPath, scanrPath, optiPath);
	const std::vector<std::vector<float>> scanFront = ds.ScanFront();
	const std::vector<std::vector<float>> scanRear = ds.ScanRear();
	const std::vector<std::vector<float>> Odoms = ds.Odom();
	const std::vector<std::vector<float>> opti = ds.OptiTrack();

	int dsFactor = 10;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), 1041, 2.268928, -2.268928);
	Lidar2D l2d_r = Lidar2D("rear_laser", Eigen::Vector3f(-0.25, -0.155, -2.356), 1041, 2.268928, -2.268928);
	std::vector<double> mask(2 * l2d_f.Heading().size(), 1.0);
	std::vector<double> scanMask = Downsample(mask, dsFactor);

	std::vector<float> commandWeights{1.0f};

	int num_frames = 300;
	int end = Odoms.size() - num_frames - 1;
	Eigen::Vector3f err = Eigen::Vector3f(0, 0, 0);
	int nExp = 50;
	int cnt = 0;

	for(int t = 0; t < nExp; ++t)
	{
		int start = int(lrand48() % end) + 1;

		NMCL nmcl = NMCL(std::make_shared<FSR>(fsr), std::make_shared<BeamEnd>(be), std::make_shared<LowVarianceResampling>(lvr), n);

		std::vector<float> odom = Odoms[start - 1];
		Eigen::Vector3f prevPose = Eigen::Vector3f(odom[0], odom[1], GetYaw(odom[5], odom[6]));
		std::vector<Eigen::Vector3f> predictions;
		std::vector<Eigen::Vector3f> groundTruth;

		for(int i = start; i < start + num_frames; ++i)
		{
			std::vector<float> odom = Odoms[i];
			Eigen::Vector3f currPose = Eigen::Vector3f(odom[0], odom[1], GetYaw(odom[5], odom[6]));
			Eigen::Vector3f delta = currPose - prevPose;

			if(((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > 0.05) || (fabs(delta(2)) > 0.05))
			{
				Eigen::Vector3f u = fsr.Backward(prevPose, currPose);
				std::vector<Eigen::Vector3f> command{u};

				nmcl.Predict(command, commandWeights, noise);
				std::vector<Eigen::Vector3f> points_3d = MergeScans(scanFront[i], l2d_f, scanRear[i], l2d_r, dsFactor);
				nmcl.Correct(points_3d, scanMask);
				prevPose = currPose;

			
				SetStatistics stas = nmcl.Stats();
				Eigen::Matrix3d cov = stas.Cov();
				Eigen::Vector3d pred = stas.Mean();

				predictions.push_back(Eigen::Vector3f(pred(0), pred(1), pred(2)));
				std::vector<float> GT = opti[i];
				groundTruth.push_back(Eigen::Vector3f(GT[0], GT[1], GT[2]));
			}
			if (i % 100 == 0)
			{
				std::vector<Particle> particles = nmcl.Particles();
				//be.plotParticles(particles, std::to_string(i), false);
			}		
		}
		if (predictions.size() > 10)
		{
			err += RMSE(predictions, groundTruth);
			++cnt;
		}
		
	}
	err = err / cnt;
	std::cout << "Avg rmse: " << err(0) << ", " << err(1) << ", " << err(2) << std::endl;
	ASSERT_LT(err.norm(), 0.25);
	
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}