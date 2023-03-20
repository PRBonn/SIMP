/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLFactory.h      	          		                               #
# ##############################################################################
**/

#ifndef NMCLFACTORY
#define NMCLFACTORY

#include "ReNMCL.h"
#include <memory>


class NMCLFactory
{
public:
	
	static std::shared_ptr<ReNMCL> Create(const std::string& configPath);

	static void Dump(const std::string& configPath);


private:

};


#endif