build:
	docker build -t simpfoxy .
run:
	docker run --rm -it --init   --gpus=all   -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(CURDIR)/..:/SIMP simpfoxy