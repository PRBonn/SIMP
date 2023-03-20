/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLUnitTests.cpp             		                           		   #
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



#include "Utils.h"


#include "Analysis.h"
#include "MixedFSR.h"
#include "SetStatistics.h"
#include "Camera.h"

#include "FloorMap.h"
#include "ReNMCL.h"
#include <nlohmann/json.hpp>
#include "NMCLFactory.h"
#include "ParticleFilter.h"
#include "BuildingMap.h"
#include "SemanticOverlap.h"

std::string testPath = PROJECT_TEST_DATA_DIR + std::string("/floor/GTMap/");



TEST(TestMixedFSR, test1) {
    
    Eigen::Vector3f p1 = Eigen::Vector3f(1.2, -2.5, 0.67);
	Eigen::Vector3f p2 = Eigen::Vector3f(-1.5, 4.3, -1.03);

	MixedFSR fsr = MixedFSR();
	Eigen::Vector3f u = fsr.Backward(p1, p2);
	Eigen::Vector3f p2_comp = fsr.Forward(p1, u);
	ASSERT_EQ(p2_comp, p2);
}

TEST(TestMixedFSR, test2) {

	MixedFSR fsr = MixedFSR();
	//test against know python result
	Eigen::Vector3f u2_gt = Eigen::Vector3f(-13.755235632332337, -3.971585665576746, 0.62);
	Eigen::Vector3f u2 = fsr.Backward(Eigen::Vector3f(13.6, -6.1, -0.33), Eigen::Vector3f(-0.7, -5.4, 0.29));
	ASSERT_NEAR(u2_gt(0), u2(0), 0.001);
	ASSERT_NEAR(u2_gt(1), u2(1), 0.001);
	ASSERT_NEAR(u2_gt(2), u2(2), 0.001);
}

TEST(TestMixedFSR, test3) {

	MixedFSR mfsr = MixedFSR();
	//float choose  = drand48();
	//test against know python result
	Eigen::Vector3f p_gt = Eigen::Vector3f(13.61489781, -6.21080639, -0.31);
	std::vector<float> commandWeights{0.5f, 0.5f};

	std::vector<Eigen::Vector3f> command{Eigen::Vector3f(0.05, -0.1, 0.02), Eigen::Vector3f(0.05, -0.1, 0.02)};
	Eigen::Vector3f noisy_p = mfsr.SampleMotion(Eigen::Vector3f(13.6, -6.1, -0.33), command, commandWeights, Eigen::Vector3f(0.0, 0.0, 0.0));
	ASSERT_NEAR(noisy_p(0), p_gt(0), 0.001);
	ASSERT_NEAR(noisy_p(1), p_gt(1), 0.001);
	ASSERT_NEAR(noisy_p(2), p_gt(2), 0.001);
	
}




TEST(TestParticleFilter, test1)
{
	std::string jsonPath = testPath + "building.config";
    BuildingMap bm(jsonPath);
	ParticleFilter pf(std::make_shared<BuildingMap>(bm));
	std::vector<Particle> particles;

	int n = 10;

	std::vector<Eigen::Vector4f> initGuesses{Eigen::Vector4f(0.1, 0.1 , 0, 0), Eigen::Vector4f(-0.1, -0.1, 0, 0)};
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero(3, 3);
	std::vector<Eigen::Matrix3d> covariances{cov, cov};

	pf.InitGaussian(particles, n, initGuesses, covariances);
	ASSERT_EQ(particles.size(), n * initGuesses.size());
	ASSERT_EQ(particles[0].pose, Eigen::Vector3f(0.1, 0.1 , 0));
	ASSERT_EQ(particles[n].pose, Eigen::Vector3f(-0.1, -0.1, 0));
}

TEST(TestParticleFilter, test2)
{
	std::string jsonPath = testPath + "building.config";
    BuildingMap bm(jsonPath);
   	ParticleFilter pf(std::make_shared<BuildingMap>(bm));
	std::vector<Particle> particles;

	int n = 10;

	Eigen::Vector4f v1(0.05, 0.05 , 0, 0);
	Eigen::Vector4f v2(-0.1, -0.1, 0, 0);


	std::vector<Eigen::Vector4f> initGuesses{v1, v2};
	Eigen::Matrix3d cov;
	cov << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1;
	std::vector<Eigen::Matrix3d> covariances{cov, cov};

	pf.InitGaussian(particles, n, initGuesses, covariances);

	Eigen::Vector3f avg1 = Eigen::Vector3f::Zero();
	Eigen::Vector3f avg2 = Eigen::Vector3f::Zero();
	for(int i = 0; i < n; ++i) avg1 += particles[i].pose;

	for(int i = n; i < 2 * n; ++i) avg2 += particles[i].pose;

	avg1 = avg1 / n;
	avg2 = avg2 / n;

	// std::cout << avg1 << std::endl;
	// std::cout << avg2 << std::endl;

	ASSERT_NEAR(avg1(0), v1(0) , 0.1);
	ASSERT_NEAR(avg1(1), v1(1) , 0.1);
	ASSERT_NEAR(avg1(2), v1(2), 0.1);
	ASSERT_NEAR(avg2(0), v2(0) , 0.1);
	ASSERT_NEAR(avg2(1), v2(1) , 0.1);
	ASSERT_NEAR(avg2(2), v2(2), 0.1);
	
}

TEST(TestParticleFilter, test3)
{
    std::string jsonPath = testPath + "building.config";
    BuildingMap bm(jsonPath);
	ParticleFilter pf(std::make_shared<BuildingMap>(bm));
	std::vector<Particle> particles;


	pf.InitUniform(particles, 1);
	std::cout << particles.size() << std::endl;
	ASSERT_EQ(particles.size(), 1);
}





TEST(TestSetStatistics, test1)
{
	std::vector<Eigen::Vector3f> poses{Eigen::Vector3f(1,1,1), Eigen::Vector3f(1,1,1)};
	std::vector<double> weights {0.5, 0.5};
	Particle p1(Eigen::Vector3f(1,1,1), 0.5);
	Particle p2(Eigen::Vector3f(1,1,1), 0.5);
	std::vector<Particle> particles{p1, p2};


	SetStatistics stats = SetStatistics::ComputeParticleSetStatistics(particles);
	Eigen::Vector3d mean = stats.Mean();
	Eigen::Matrix3d cov = stats.Cov();

	ASSERT_EQ(mean, Eigen::Vector3d(1,1,1));

	ASSERT_NEAR(cov(0,0), 0, 0.000001);
	ASSERT_NEAR(cov(1,0), 0, 0.000001);
	ASSERT_NEAR(cov(0,1), 0, 0.000001);
	ASSERT_NEAR(cov(1,1), 0, 0.000001);
	ASSERT_NEAR(cov(2,2), 0, 0.000001);

}

TEST(TestSetStatistics, test2)
{
	std::vector<Eigen::Vector3f> poses{Eigen::Vector3f(1.3 ,1 ,1), Eigen::Vector3f(0.8, 0.7, 0)};
	std::vector<double> weights {0.5, 0.5};
	Particle p1(Eigen::Vector3f(1.3,1,1), 0.5);
	Particle p2(Eigen::Vector3f(0.8,0.7,0), 0.5);
	std::vector<Particle> particles{p1, p2};

	SetStatistics stats = SetStatistics::ComputeParticleSetStatistics(particles);
	Eigen::Vector3d mean = stats.Mean();
	Eigen::Matrix3d cov = stats.Cov();

	//ASSERT_EQ(mean, Eigen::Vector3d(1.05, 0.85, 0.5));
	ASSERT_NEAR(mean(0), 1.05 , 0.000001);
	ASSERT_NEAR(mean(1), 0.85 , 0.000001);
	ASSERT_NEAR(mean(2), 0.5 , 0.000001);
	ASSERT_NEAR(cov(0,0), 0.0625 , 0.000001);
	ASSERT_NEAR(cov(1,0), 0.0375 , 0.000001);
	ASSERT_NEAR(cov(0,1), 0.0375 , 0.000001);
	ASSERT_NEAR(cov(1,1), 0.0225, 0.000001);
	ASSERT_GE(cov(2,2), 0);  

}

TEST(TestSetStatistics, test3)
{
	std::vector<Eigen::Vector3f> poses{Eigen::Vector3f(1 ,1 ,1), Eigen::Vector3f(0, 0, 0)};
	std::vector<double> weights {1.0, 0.0};
	Particle p1(Eigen::Vector3f(1, 1,1 ), 1.0);
	Particle p2(Eigen::Vector3f(0, 0, 0), 0.0);
	std::vector<Particle> particles{p1, p2};

	SetStatistics stats = SetStatistics::ComputeParticleSetStatistics(particles);
	Eigen::Vector3d mean = stats.Mean();
	Eigen::Matrix3d cov = stats.Cov();

	ASSERT_EQ(mean, Eigen::Vector3d(1, 1, 1));

	ASSERT_NEAR(cov(0,0), 0, 0.000001);
	ASSERT_NEAR(cov(1,0), 0, 0.000001);
	ASSERT_NEAR(cov(0,1), 0, 0.000001);
	ASSERT_NEAR(cov(1,1), 0, 0.000001);
	ASSERT_NEAR(cov(2,2), 0, 0.000001);

}



TEST(TestNMCLFactory, test1)
{
	std::string configPath = testPath + "nmcl.config";
	//NMCLFactory::Dump(configPath);

	std::shared_ptr<ReNMCL> renmcl = NMCLFactory::Create(configPath);
	std::string roonName = renmcl->GetFloorMap()->GetRoomNames()[1];
	//std::cout << renmcl->GetFloorMap()->GetRoomNames()[1] << std::endl;

	ASSERT_EQ(roonName, "0");
}






int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}