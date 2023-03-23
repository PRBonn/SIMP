
import json

class Omni3DImage():

	def __init__(self, id_, dataset_id, seq_id, width, height, file_path, K, gt, t, src_90_rotate=0, src_flagged=0):

		self.id = id_
		self.dataset_id = dataset_id
		self.seq_id = seq_id
		self.width = width
		self.height = height
		self.file_path = file_path
		self.K = K
		self.src_90_rotate = src_90_rotate
		self.src_flagged = src_flagged
		self.gt = gt
		self.t = t

	def toJSON(self):
		 return json.dumps(self, default=vars, indent=4)
		