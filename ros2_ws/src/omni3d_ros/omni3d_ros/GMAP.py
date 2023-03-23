
import numpy as np
import cv2
import yaml
import os 

class GMAP():



    def __init__(self, *args):


        if len(args) == 1:
            yamlPath = args[0]
            folderPath, _ = os.path.split(yamlPath)
            print(folderPath)
            with open(yamlPath, 'r') as stream:
                data = yaml.safe_load(stream)

            self.map = cv2.imread(folderPath + "/" + data['image'])
            self.resolution = data['resolution']
            self.origin = data['origin']
            self.max_y = self.map.shape[0]
            self.ComputeBorders()

        elif len(args) == 2:
            yamlConf = args[0]
            gridmap = args[1]
            self.map = gridmap
            self.resolution = yamlConf['resolution']
            self.origin = yamlConf['origin']
            self.max_y = gridmap.shape[0]
            self.ComputeBorders()

        elif len(args) == 3:
            gridmap = args[0]
            resolution = args[1]
            origin = args[2]
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
