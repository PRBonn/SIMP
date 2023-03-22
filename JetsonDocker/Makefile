build:
	docker build -t cubercnn .
run:
	docker run --runtime nvidia -it --rm --network host -e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix/:/tmp/.X11-unix \
	-v $(CURDIR)/../omni3d_ws:/omni3d_ws \
	-e  ROS_IP=192.168.131.69 \
	-e ROS_MASTER_URI=http://192.168.131.1:11311 \
	-e ROS_HOSTNAME=jetson \
	cubercnn
