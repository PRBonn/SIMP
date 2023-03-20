cmake_minimum_required(VERSION 3.16.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nsensors/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann_json/include/)

set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)



add_subdirectory("src")
#add_library(NSENSORS STATIC IMPORTED)
#set_target_properties(NSENSORS PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libNSENSORS.a)


# After all are setup is done, we can go to our src/ directory to build our
# files
if(BUILD_TESTING)
  add_subdirectory("tst")
endif()
