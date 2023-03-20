

add_executable(NMapUnitTests NMapUnitTests.cpp )
target_link_libraries(NMapUnitTests ${OpenCV_LIBS} ${catkin_LIBRARIES} NMAP NSENSORS GTest::GTest gtest_main ${Boost_LIBRARIES})
target_compile_definitions(NMapUnitTests PRIVATE PROJECT_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data/")
add_test(AllTestsInTests NMapUnitTests)
