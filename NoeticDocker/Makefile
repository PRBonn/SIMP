build:
	docker build -t simpnoetic .
run:
	docker run --rm -it --init   --gpus=all   -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(CURDIR)/../ros1_ws/data:/packages/data simpnoetic