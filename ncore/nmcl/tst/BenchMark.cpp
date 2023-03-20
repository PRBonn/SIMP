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

#include "BeamEnd.h"
#include "NMCL.h"
#include "Dataset.h"
#include "Analysis.h"
#include "MixedFSR.h"
#include "SetStatistics.h"
#include "VODataSet.h"
#include "MixedFSR.h"
#include "LowVarianceResampling.h"


std::string dataPath = PROJECT_TEST_DATA_DIR + std::string("/21/");


void TestVONMCL(VODataSet& vods)
{
	int n = 10000;

	std::shared_ptr<OptiTrack> op = std::make_shared<OptiTrack>(OptiTrack(dataPath));

	GMap gmap = GMap(dataPath);
	LowVarianceResampling lvr = LowVarianceResampling();


	BeamEnd be = BeamEnd(std::make_shared<GMap>(gmap), 8, 15);
	MixedFSR fsr = MixedFSR();
	Eigen::Vector3f noise = Eigen::Vector3f(0.1, 0.1, 0.1);

	int dsFactor = 10;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), 1041, 2.268928, -2.268928);
	Lidar2D l2d_r = Lidar2D("rear_laser", Eigen::Vector3f(-0.25, -0.155, -2.356), 1041, 2.268928, -2.268928);
	std::vector<double> mask(2 * l2d_f.Heading().size(), 1.0);
	std::vector<double> scanMask = Downsample(mask, dsFactor);

	std::vector<float> commandWeights{0.5f, 0.5f};

	int num_frames = 300;
	int end = vods.FrameNum() - num_frames - 1;
	Eigen::Vector3f err = Eigen::Vector3f(0, 0, 0);
	int nExp = 50;
	int cnt = 0;

	int numBeams = vods.BeamNum();

	for(int t = 0; t < nExp; ++t)
	{
		int start = int(lrand48() % end) + 1;

		//NMCL vonmcl = NMCL(std::make_shared<MixedFSR>(fsr), std::make_shared<BeamEnd>(be), n);
		NMCL vonmcl = NMCL(std::make_shared<FSR>(fsr), std::make_shared<BeamEnd>(be), std::make_shared<LowVarianceResampling>(lvr), n);



		Frame frame = vods.ReadFrame(start - 1);
		float* wo = frame.WO; 
		float* vo = frame.VO; 

		Eigen::Vector3f prevWOPose = Eigen::Vector3f(wo[0], wo[1], GetYaw(wo[5], wo[6]));
		Eigen::Vector3f prevVOPose = Eigen::Vector3f(vo[0], vo[1], GetYaw(vo[5], vo[6]));

		std::vector<Eigen::Vector3f> predictions;
		std::vector<Eigen::Vector3f> groundTruth;

		for(int i = start; i < start + num_frames; ++i)
		{

			Frame frame = vods.ReadFrame(i);
			float* wo = frame.WO; 
			float* vo = frame.VO; 

			Eigen::Vector3f currWOPose = Eigen::Vector3f(wo[0], wo[1], GetYaw(wo[5], wo[6]));
			Eigen::Vector3f currVOPose = Eigen::Vector3f(vo[0], vo[1], GetYaw(vo[5], vo[6]));
			Eigen::Vector3f delta = currWOPose - prevWOPose;



			if(((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > 0.05) || (fabs(delta(2)) > 0.05))
			{
				Eigen::Vector3f uVO = fsr.Backward(prevVOPose, currVOPose);
				Eigen::Vector3f uWO = fsr.Backward(prevWOPose, currWOPose);
				std::vector<Eigen::Vector3f> command{uVO, uWO};

				float* frnt = frame.scanFront; 
				std::vector<float> scanFront {frnt, frnt + numBeams};

				float* rear = frame.scanRear; 
				std::vector<float> scanRear {rear, rear + numBeams};

				float* opti = frame.GT; 
				Eigen::Vector3f gt = op->OptiTrack2World(Eigen::Vector3f(opti[0], opti[1], GetYaw(opti[5], opti[6])));

				vonmcl.Predict(command, commandWeights, noise);
				std::vector<Eigen::Vector3f> points_3d = MergeScans(scanFront, l2d_f, scanRear, l2d_r, dsFactor);
				vonmcl.Correct(points_3d, scanMask);

				prevWOPose = currWOPose;
				prevVOPose = currVOPose;

			
				SetStatistics stas = vonmcl.Stats();
				Eigen::Matrix3d cov = stas.Cov();
				Eigen::Vector3d pred = stas.Mean();
				//std::cout << pred << std::endl;

				if(pred.array().isNaN().any())
				{
					std::cout << "Localization failed!" << std::endl;
					break;
				}
				else
				{
					predictions.push_back(Eigen::Vector3f(pred(0), pred(1), pred(2)));
					groundTruth.push_back(gt);
				}
			}
			if (i % 100 == 0)
			{
				std::vector<Particle> particles = vonmcl.Particles();
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

void TestNMCL(VODataSet& vods)
{
	int n = 10000;

	std::shared_ptr<OptiTrack> op = std::make_shared<OptiTrack>(OptiTrack(dataPath));

	GMap gmap = GMap(dataPath);
	BeamEnd be = BeamEnd(std::make_shared<GMap>(gmap));
	FSR fsr = FSR();
	Eigen::Vector3f noise = Eigen::Vector3f(0.1, 0.1, 0.1);
	LowVarianceResampling lvr = LowVarianceResampling();

	int dsFactor = 10;
	Lidar2D l2d_f = Lidar2D("front_laser", Eigen::Vector3f(0.25, 0.155, 0.785), 1041, 2.268928, -2.268928);
	Lidar2D l2d_r = Lidar2D("rear_laser", Eigen::Vector3f(-0.25, -0.155, -2.356), 1041, 2.268928, -2.268928);
	std::vector<double> mask(2 * l2d_f.Heading().size(), 1.0);
	std::vector<double> scanMask = Downsample(mask, dsFactor);

	std::vector<float> commandWeights{1.0f};

	int num_frames = 300;
	int end = vods.FrameNum() - num_frames - 1;
	Eigen::Vector3f err = Eigen::Vector3f(0, 0, 0);
	int nExp = 50;
	int cnt = 0;

	int numBeams = vods.BeamNum();

	for(int t = 0; t < nExp; ++t)
	{
		int start = int(lrand48() % end) + 1;

		//NMCL nmcl = NMCL(std::make_shared<FSR>(fsr), std::make_shared<BeamEnd>(be), n);
		NMCL nmcl = NMCL(std::make_shared<FSR>(fsr), std::make_shared<BeamEnd>(be), std::make_shared<LowVarianceResampling>(lvr), n);


		Frame frame = vods.ReadFrame(start - 1);
		float* wo = frame.WO; 
		
		Eigen::Vector3f prevWOPose = Eigen::Vector3f(wo[0], wo[1], GetYaw(wo[5], wo[6]));
	
		std::vector<Eigen::Vector3f> predictions;
		std::vector<Eigen::Vector3f> groundTruth;

		for(int i = start; i < start + num_frames; ++i)
		{

			Frame frame = vods.ReadFrame(i);
			float* wo = frame.WO; 

			Eigen::Vector3f currWOPose = Eigen::Vector3f(wo[0], wo[1], GetYaw(wo[5], wo[6]));
			Eigen::Vector3f delta = currWOPose - prevWOPose;



			if(((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > 0.05) || (fabs(delta(2)) > 0.05))
			{
				Eigen::Vector3f uWO = fsr.Backward(prevWOPose, currWOPose);
				std::vector<Eigen::Vector3f> command{uWO};

				float* frnt = frame.scanFront; 
				std::vector<float> scanFront {frnt, frnt + numBeams};

				float* rear = frame.scanRear; 
				std::vector<float> scanRear {rear, rear + numBeams};

				float* opti = frame.GT; 
				Eigen::Vector3f gt = op->OptiTrack2World(Eigen::Vector3f(opti[0], opti[1], GetYaw(opti[5], opti[6])));

				nmcl.Predict(command, commandWeights, noise);
				std::vector<Eigen::Vector3f> points_3d = MergeScans(scanFront, l2d_f, scanRear, l2d_r, dsFactor);
				nmcl.Correct(points_3d, scanMask);

				prevWOPose = currWOPose;
						
				SetStatistics stas = nmcl.Stats();
				Eigen::Matrix3d cov = stas.Cov();
				Eigen::Vector3d pred = stas.Mean();
				if(pred.array().isNaN().any())
				{
					std::cout << "Localization failed!" << std::endl;
				}
				else
				{
					predictions.push_back(Eigen::Vector3f(pred(0), pred(1), pred(2)));
					groundTruth.push_back(gt);
				}
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





int main()
{
	VODataSet vods;
	//vods.LoadDataSet(std::string(dataPath + "TestRun2.txt"));
	vods.LoadDataSet(std::string(dataPath + "Legs.txt"));
	TestVONMCL(vods);
	TestNMCL(vods);

	return 0;
}