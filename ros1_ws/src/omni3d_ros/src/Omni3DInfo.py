import json

class Omni3DInfo():
	def __init__(self, id_, source, name, split, version, url):
		self.id = id_
		self.source = source
		self.name = name
		self.split = split
		self.version = version
		self.url = url


	def toJSON(self):
		 return json.dumps(self, default=vars, sort_keys=True, indent=4)
		