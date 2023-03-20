


add_executable(NMCLUnitTests NMCLUnitTests.cpp Analysis.cpp)
target_link_libraries(NMCLUnitTests ${OpenCV_LIBS} ${catkin_LIBRARIES} NMCL NSENSORS NMAP nlohmann_json::nlohmann_json GTest::GTest gtest_main ${Boost_LIBRARIES})
target_compile_definitions(NMCLUnitTests PRIVATE PROJECT_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
add_test(AllTestsInTests NMCLUnitTests)


