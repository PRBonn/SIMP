

import json


class Omni3DObject():

	uid = 0


	def __init__(self, dataset_id, image_id, seq_id, category_id, category_name, \
				valid3D, bbox2D_tight, bbox2D_proj, bbox2D_trunc, bbox3D_cam, center_cam, dimensions, \
				R_cam, behind_camera=-1, visibility=-1, truncation=-1, segmentation_pts=-1, lidar_pts=-1, depth_error=-1):

		self.id = Omni3DObject.uid
		Omni3DObject.uid += 1
		self.dataset_id = dataset_id
		self.image_id = image_id
		self.seq_id = seq_id
		self.category_id = category_id
		self.category_name = category_name
		self.valid3D = valid3D
		self.bbox2D_tight = bbox2D_tight
		self.bbox2D_proj = bbox2D_proj
		self.bbox2D_trunc = bbox2D_trunc
		self.bbox3D_cam = bbox3D_cam
		self.center_cam = center_cam
		self.dimensions = dimensions
		self.R_cam = R_cam
		self.behind_camera = behind_camera
		self.visibility = visibility
		self.truncation = truncation
		self.segmentation_pts = segmentation_pts
		self.lidar_pts = lidar_pts
		self.depth_error = depth_error

	def toJSON(self):
		 return json.dumps(self, default=vars, sort_keys=True, indent=4)
