#add_executable(RoomSegmentation RoomSegmentation.cpp GMap.cpp FloorMap.cpp Room.cpp Lift.cpp Object.cpp BuildingMap.cpp)
#target_link_libraries(RoomSegmentation ${OpenCV_LIBS} NSENSORS ${Boost_LIBRARIES})
add_library(NMAP GMap.cpp BuildingMap.cpp FloorMap.cpp Room.cpp Lift.cpp Object.cpp)



