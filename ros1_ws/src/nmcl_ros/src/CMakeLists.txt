






add_executable(ConfigNMCLNode ConfigNMCLNode.cpp RosUtils.cpp)
target_link_libraries(ConfigNMCLNode ${OpenCV_LIBS} ${catkin_LIBRARIES} NMCL NSENSORS NMAP ${Boost_LIBRARIES} nlohmann_json::nlohmann_json)

