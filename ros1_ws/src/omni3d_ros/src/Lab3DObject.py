
import numpy as np



class Lab3DObject():

	gid = 0

	def __init__(self, oid, category, pos, dim, yaw):
		self.oid = oid
		self.category = category
		self.pos = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64) 
		self.dim = np.array([dim['x'], dim['y'], dim['z']], dtype=np.float64)
		self.yaw = yaw
		self.gid = Lab3DObject.gid

		Lab3DObject.gid += 1


	def __init__(self, jObj):

		self.oid = jObj['id']
		self.category = jObj['category_id']
		pos = jObj['position']
		dim = jObj['dimensions']
		self.pos = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64) 
		self.dim = np.array([dim['x'], dim['y'], dim['z']], dtype=np.float64)
		self.yaw = jObj['yaw']
		self.gid = Lab3DObject.gid

		Lab3DObject.gid += 1