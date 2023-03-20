cmake_minimum_required(VERSION 3.16.0)

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann_json/include/)

set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)


# After all are setup is done, we can go to our src/ directory to build our
# files
add_subdirectory("src")
if(BUILD_TESTING)
  add_subdirectory("tst")
endif()
