build:
	docker build -f FoxyDockerfile -t simpfoxy .
run:
	docker run --rm -it --init   --gpus=all  --network host -e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	 -v $(CURDIR):/SIMP simpfoxy