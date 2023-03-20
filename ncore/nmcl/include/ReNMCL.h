/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ReNMCL.h      	          				                               #
# ##############################################################################
**/

#ifndef RENMCL_H
#define RENMCL_H

#include "MixedFSR.h"
#include "Resampling.h"
#include "SetStatistics.h"
#include "FloorMap.h"
#include <memory>
#include "ParticleFilter.h"
#include "SemanticOverlap.h"
#include "BuildingMap.h"
#include "Semantic3DData.h"

class ReNMCL
{
	public:


		enum class Strategy 
		{   
			UNIFORM = 0, 
		    BYROOM = 1
		};

		//! A constructor
	    /*!
	     \param fm is a ptr to a FloorMap object
	      \param mm is a ptr to a MotionModel object, which is an abstract class. FSR is the implementation 
	      \param sm is a ptr to a BeamEnd object, which is an abstract class. BeamEnd is the implementation 
	      \param rs is a ptr to a Resampling object, which is an abstract class. LowVarianceResampling is the implementation 
	      \param n_particles is an int, and it defines how many particles the particle filter will use
	      \param injectionRatio is an float, and it determines which portions of the particles are replaced when relocalizing
	    */
		ReNMCL(std::shared_ptr<BuildingMap> bm,  std::shared_ptr<Resampling> rs, std::shared_ptr<MixedFSR> mm, std::shared_ptr<SemanticOverlap> sem3d, int n_particles);


		//! A constructor
	    /*!
	     * \param fm is a ptr to a FloorMap object
	      \param mm is a ptr to a MotionModel object, which is an abstract class. FSR is the implementation 
	      \param sm is a ptr to a BeamEnd object, which is an abstract class. BeamEnd is the implementation 
	      \param rs is a ptr to a Resampling object, which is an abstract class. LowVarianceResampling is the implementation 
	      \param n_particles is an int, and it defines how many particles the particle filter will use
	      \param initGuess is a vector of initial guess for the location of the robots
	      \param covariances is a vector of covariances (uncertainties) corresponding to the initial guesses
	      \param injectionRatio is an float, and it determines which portions of the particles are replaced when relocalizing
	    */
		ReNMCL(std::shared_ptr<BuildingMap> bm,  std::shared_ptr<Resampling> rs, std::shared_ptr<MixedFSR> mm, std::shared_ptr<SemanticOverlap> sem3d, int n_particles, 
			std::vector<Eigen::Vector4f> initGuess, 
			std::vector<Eigen::Matrix3d> covariances);


		//! A getter for the mean and covariance of the particles
		/*!
		   \return an object SetStatistics, that has fields for mean and covariance
		*/
		SetStatistics Stats()
		{
			return o_stats;
		}


		//! A getter particles representing the pose hypotheses 
		/*!
		   \return A vector of points, where each is Eigen::Vector3f = (x, y, theta)
		*/
		std::vector<Particle> Particles()
		{
			return o_particles;
		}


		//! Advanced all particles according to the control and noise, using the chosen MotionModel's forward function
		/*!
		  \param control is a 3d control command. In the FSR model it's (forward, sideways, rotation)
		  \param odomWeights is the corresponding weight to each odometry source
	      \param noise is the corresponding noise to each control component
		*/
		void Predict(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);

	

		//! Considers the semantic likelihood of observation for all hypotheses, and then performs resampling. 
		/*!
		  \param data is an SemanticData ptr that hold the labels and center locations for all detected objects. The center locations are in the base_link frame
		*/

		void CorrectOmni(std::shared_ptr<Semantic3DData> data);


		//! Initializes filter with new particles upon localization failure
		void Recover();

		Eigen::Vector3f Backward(Eigen::Vector3f p1, Eigen::Vector3f p2)
		{
			return o_motionModel->Backward(p1, p2);
		}


		const std::shared_ptr<FloorMap>& GetFloorMap()
		{
			return o_floorMap;
		}


		void SetPredictStrategy(Strategy strategy)
		{
			o_predictStrategy = strategy;
		}



	private:

		void predictUniform(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);

		void dumpParticles();
	
		Strategy o_predictStrategy = Strategy(0);
		std::shared_ptr<SemanticOverlap> o_semanticModel3;
		std::shared_ptr<ParticleFilter> o_particleFilter;
		unsigned int o_numFloors;
		unsigned int o_frame = 0;

		std::shared_ptr<MixedFSR> o_motionModel;
		std::shared_ptr<Resampling> o_resampler; 
		std::shared_ptr<FloorMap> o_floorMap;
		std::shared_ptr<BuildingMap> o_buildingMap;
		std::vector<std::shared_ptr<GMap>> o_gmaps;

		int o_numParticles = 0;
		std::vector<Particle> o_particles;
		SetStatistics o_stats;
};

#endif