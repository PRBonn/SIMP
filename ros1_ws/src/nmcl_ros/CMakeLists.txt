cmake_minimum_required(VERSION 3.0.2)
project(nmcl_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

#set(CMAKE_CXX_FLAGS "-std=c++11 -O3  -Wall ${CMAKE_CXX_FLAGS}")

#set(nlohmann_json_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/nlohmann_json)


## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nmcl/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nsensors/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nmap/include/)


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/nmcl.cpp
# )

set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib)


# Attempt to find OpenCV4 on your system, for more details please read
# /usr/share/OpenCV/OpenCVConfig.cmake

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(
  catkin REQUIRED COMPONENTS roscpp sensor_msgs message_filters geometry_msgs nav_msgs tf2 tf cv_bridge pcl_ros nmcl_msgs)
include_directories(${catkin_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
  message(STATUS "OpenCV directories: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "OpenCV not found, please read the README.md")
endif(OpenCV_FOUND)

find_package(Boost COMPONENTS serialization system filesystem)
if(Boost_FOUND)
  message(STATUS "Found Boost version ${Boost_VERSION}")
  message(STATUS "Boost directories: ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Boost not found, please read the README.md")
endif(Boost_FOUND)

#find_package(Torch REQUIRED)
#if(Torch_FOUND)
#  message(STATUS "Found Torch version ${Torch_VERSION}")
#  message(STATUS "Torch directories: ${Torch_INCLUDE_DIRS}")
#  include_directories(${Torch_INCLUDE_DIRS})
#else()
#  message(FATAL_ERROR "OpenCV not found, please read the README.md")
#endif(Torch_FOUND)

find_package(nlohmann_json 3.2.0 REQUIRED)
if(nlohmann_json_FOUND)
  message(STATUS "Found nlohmann_json")
else()
  message(FATAL_ERROR "nlohmann_json not found, please read the README.md")
endif(nlohmann_json_FOUND)

# After all are setup is done, we can go to our src/ directory to build our
# files
add_subdirectory("src")
message(STATUS "Nicky ${CMAKE_CURRENT_SOURCE_DIR}")
add_library(NMCL STATIC IMPORTED)
add_library(NSENSORS STATIC IMPORTED)
add_library(NMAP STATIC IMPORTED)


# Provide the full path to the library, so CMake knows where to find it.
set_target_properties(NMCL PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNMCL.a)
set_target_properties(NSENSORS PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNSENSORS.a)
set_target_properties(NMAP PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNMAP.a)
