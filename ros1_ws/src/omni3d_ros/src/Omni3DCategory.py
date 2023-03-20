
import json

class Omni3DCategory():

	def __init__(self, id_, name, supercategory=""):

		self.id = id_
		self.name = name
		self.supercategory = supercategory

	def toJSON(self):
		 return json.dumps(self, default=vars, sort_keys=True, indent=4)