import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


for a in range(4):

	df = pd.read_csv("/home/nickybones/Code/YouBotMCL/ncore/build/" + str(90 * a) + "particles.csv") 
	x = df.iloc[:, 0]
	y = df.iloc[:, 1]
	yaw = df.iloc[:, 2]

	plt.gca().invert_yaxis()

	plt.scatter(x, y, c='k')
	plt.scatter([0], [0], c='r')

	for i in range(len(yaw)):
		ang = yaw[i]
		dx = np.cos(ang)
		dy = np.sin(ang)
		plt.arrow(x[i], y[i], 0.005*dx, 0.005*dy) 

	plt.savefig("/home/nickybones/Code/YouBotMCL/ncore/build/" + str(90 * a) + "particles.png")
	plt.close()
	#plt.show()



