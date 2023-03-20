#!/usr/bin/env python3

import cv2
import rospy
import pandas as pd
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped, Transform
import tf
from tf import transformations as ts
import numpy as np
import math
from scipy.interpolate import interp1d
from bisect import insort, bisect_left
from collections import deque
from itertools import islice
from scipy.spatial.transform import Rotation as R

camera_z = 0.63
min_z = -1.55708286

def running_median_insort(seq, window_size):
	"""Contributed by Peter Otten"""
	seq = iter(seq)
	d = deque()
	s = []
	result = []
	for item in islice(seq, window_size):
	    d.append(item)
	    insort(s, item)
	    result.append(s[len(d)//2])
	m = window_size // 2
	for item in seq:
	    old = d.popleft()
	    d.append(item)
	    del s[bisect_left(s, old)]
	    insort(s, item)
	    result.append(s[m])
	return result


def interpolate(t, poses, kind='nearest', window=2, tol=0.5):
	"""Contributed by Jerome Guzzi and adapted to our data"""

	n = len(poses)
	data = np.zeros((n, 4))

	for i in range(n):
	  data[i, 0] = t[i]
	  data[i, 1:] = poses[i]


	# Filter outliers in yaw
	data = data[np.abs(data[:, 3] - running_median_insort(data[:, 3], window)) < tol]
	data[0, -1] = np.fmod(data[0, -1], 2 * np.pi)

	return interp1d(data[:, 0], data[:, 1:], axis=0, fill_value='extrapolate', assume_sorted=True, kind=kind)




def main():
    

	df = pd.read_csv("/home/nickybones/data/MCL/omni3d/Map3/raw_mcl.csv")
	gt_t = []
	gt_poses = []
	n = len(df['gt_x'])
	t = df['t']
	x = df['gt_x'].to_numpy()
	y = df['gt_y'].to_numpy()
	yaw = df['gt_yaw'].to_numpy()


	for i in range(n):

		gt_poses.append(np.array([x[i], y[i], yaw[i]]))
		gt_t.append(t[i])
		#print(gt_poses[i], gt_t[i])


	t1 = 1673979858458452341
	t2 = 1673979858505056416
	t3 = int((t1 + t2) / 2)
	print("[ 7.02680039 -2.95074724  3.90965687] 1673979858458452341")
	print("[ 7.02663729 -2.94933241  3.84571609] 1673979858505056416")
	intr = interpolate(gt_t, gt_poses)


	pose = intr(t3)
	print(pose, t3)

if __name__ == "__main__":

    main()