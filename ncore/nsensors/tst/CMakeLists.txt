

add_executable(NSensorsUnitTests NSensorsUnitTests.cpp )
target_link_libraries(NSensorsUnitTests ${OpenCV_LIBS} ${catkin_LIBRARIES} NSENSORS GTest::GTest gtest_main)
target_compile_definitions(NSensorsUnitTests PRIVATE PROJECT_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data/")
add_test(AllTestsInTests NSensorsUnitTests)
