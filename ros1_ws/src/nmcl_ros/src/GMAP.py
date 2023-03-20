import numpy as np
import cv2

class GMAP():

    def __init__(self, yaml, gridmap):

        self.map = gridmap
        self.resolution = yaml['resolution']
        self.origin = yaml['origin']
        self.max_y = gridmap.shape[0]
        self.ComputeBorders()


    def __init__(self, gridmap, resolution, origin):

        self.map = gridmap
        self.resolution = resolution
        self.origin = origin
        self.max_y = gridmap.shape[0]
        self.ComputeBorders()

    # the (0,0) of the gmapping map is where the robot started from. To relate this to the gridmap image
    # We need to know the real world coordinates of some image point.
    # the map origin in the bottom left corner of the image, and its real world coordinates are
    # specified in the metadata yaml.

    def world2map(self, p):
        u = np.round((p[0] - self.origin[0]) / self.resolution)
        v = self.max_y - np.round((p[1] - self.origin[1]) / self.resolution)
        return np.array([u, v]).astype(int)

    def map2world(self, uv):
        x = uv[0] * self.resolution + self.origin[0]
        y = (self.max_y - uv[1]) * self.resolution + self.origin[1]

        return np.array([x, y])

    def ComputeBorders(self):
        self.map = 255 - cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
        white_pixels = np.array(np.where(self.map == 255))
        min_x = min(white_pixels[1])
        max_x = max(white_pixels[1])
        min_y = min(white_pixels[0])
        max_y = max(white_pixels[0])
        self.map_border = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}

    def TopLeft(self):
        return np.array([self.map_border["min_x"], self.map_border["min_y"]])

    def BottomRight(self):
        return np.array([self.map_border["max_x"], self.map_border["max_y"]])
