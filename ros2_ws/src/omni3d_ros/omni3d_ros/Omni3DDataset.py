import json

class Omni3DDataset():

	def __init__(self, info, images, categories, annotations):
		self.info = info
		self.images = images
		self.categories = categories
		self.annotations = annotations


	def toJSON(self):
		 return json.dumps(self, default=vars, sort_keys=True, indent=4)
		