cmake_minimum_required(VERSION 3.5)
project(nmcl_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nmcl/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nsensors/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/nmap/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/ndl/include/)


set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib)

# find dependencies
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_library(NMCL STATIC IMPORTED)
add_library(NSENSORS STATIC IMPORTED)
add_library(NMAP STATIC IMPORTED)
add_library(NDL STATIC IMPORTED)
set_target_properties(NMCL PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNMCL.a)
set_target_properties(NSENSORS PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNSENSORS.a)
set_target_properties(NMAP PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNMAP.a)
set_target_properties(NDL PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../ncore/build/lib/libNDL.a)

add_executable(ScanMergeNode src/ScanMergeNode.cpp src/RosUtils.cpp)
target_link_libraries(ScanMergeNode NSENSORS ${Boost_SYSTEM_LIBRARY})
ament_target_dependencies(ScanMergeNode rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs message_filters tf2 tf2_ros pcl_ros pcl_conversions OpenCV)

add_executable(ConfigNMCLNode src/ConfigNMCLNode.cpp src/RosUtils.cpp)
target_link_libraries(ConfigNMCLNode NMCL NMAP NSENSORS ${Boost_SYSTEM_LIBRARY})
ament_target_dependencies(ConfigNMCLNode rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs message_filters tf2 tf2_ros pcl_ros pcl_conversions nlohmann_json OpenCV)



install(DIRECTORY
  launch
 # config
 # rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
