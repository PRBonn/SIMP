build:
ifdef  foxy
	docker build -f FoxyDockerfile -t simpfoxy .
endif
ifdef noetic
	docker build -f NoeticDockerfile -t simpnoetic .
endif
ifdef  jetson
	docker build -f JetsonDockerfile -t simpjetson .
endif
run:
ifdef  foxy
	docker run --rm -it --init   --gpus=all  --network host -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	 -v $(CURDIR):/SIMP simpfoxy
endif
ifdef noetic
	docker run --rm -it --init   --gpus=all  --network host -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	 -v $(CURDIR):/SIMP simpnoetic
endif
ifdef jetson
	docker run --rm -it --init   --gpus=all  --network host -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-e  ROS_IP=192.168.131.69 \
	-e ROS_MASTER_URI=http://192.168.131.1:11311 \
	-e ROS_HOSTNAME=jetson \
	-v $(CURDIR):/SIMP simpjetson
endif