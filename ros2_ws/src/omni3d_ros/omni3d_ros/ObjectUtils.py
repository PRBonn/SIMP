

from MapObjectTracker import *
import pandas as pd 
import numpy as np
from DatasetUtils import *

def DumpTrackedObjectsToPickle(output_dir, mapObjects):

	category = []
	center = []
	dim = []
	rot = []
	skip = []
	times_matched = []
	skip = []
	uid = []
	conf = []
	times_skipped = []
	rooms = []

	for o in mapObjects:
		category.append(o.category)
		center.append(o.center.flatten())
		dim.append(o.dim)
		rot.append(o.rot.flatten())
		skip.append(o.skip)
		times_matched.append(o.times_matched)
		uid.append(o.uid)
		conf.append(o.conf)
		times_skipped.append(o.times_skipped)
		rooms.append(o.room)

	print(len(category), len(center), len(dim), len(rot), len(skip), len(times_matched), len(uid), len(conf))
	#datadic = {'category': category, 'center': center, 'dim' : dim, 'rot': rot, 'skip': skip, 'times_matched' : times_matched, 'conf' : conf, 'uid' : uid, 'times_skipped': times_skipped}
	#datadic = {'category': category, 'center': center, 'dim' : dim, 'rot': rot, 'skip': skip, 'times_matched' : times_matched, 'conf' : conf, 'uid' : uid, 'times_skipped': times_skipped}
	datadic = {'category': category, 'center': center, 'dim' : dim, 'rot': rot, 'skip': skip, 'times_matched' : times_matched, 'conf' : conf, 'uid' : uid, 'times_skipped': times_skipped, "room": rooms}

	df = pd.DataFrame(datadic)
	df.to_pickle(output_dir + "/mapObjects.pickle")   


def LoadTrackedObjectsFromPickle(dataDir, name="/mapObjects.pickle"):		

	df = pd.read_pickle(dataDir + name)
	mapObjects = []

	for index, row in df.iterrows(): 
		rot = np.reshape(row['rot'], (3,3))
		#print(rot)
		obj = MapObjectTracker(row['category'], row['center'], row['dim'], rot, row['conf'], room=row['room'], static=False, skip=row['skip'], times_matched=row['times_matched'], times_skipped=row['times_skipped'] )
		mapObjects.append(obj)

	return mapObjects


def getBoxFromObj(obj):
	center = obj.center
	dim = obj.dim
	rot = np.reshape(obj.rot, (3,3))
	box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
	verts, faces = get_cuboid_verts_faces(box3d, rot)
	verts = torch.unsqueeze(verts, dim=0)
	return verts


def FindAllObjectsByCategoty(objects, category):

	matches = []

	for o in objects:
		if o.category == category:
			matches.append(o)


	return matches

def JSONObjects2MapObject(o):

	mobj = MapObjectTracker(o['category_id'], o['center_cam'], o['dimensions'], np.reshape(o['R_cam'], (3,3)), o['score'])

	return mobj


def loadGTObjects(origin, jsonPath, roomNum=None):

	with open(jsonPath, 'r') as f:
	    data = json.load(f)

	objs = []
	uid = 0

	rooms = data['dataset']['samples']
	if roomNum:

		for r in range(len(rooms)):
			roomName = rooms[r]['name'].replace(".pcd", "")

			if roomNum == 11:
				if "Room{}".format(roomNum) in roomName:
					annos = rooms[r]['labels']['ground-truth']['attributes']['annotations']
					for a in annos:
						category = a['category_id']
						pos = a['position']
						dim = a['dimensions']
						dimensions = np.array([dim['x'], dim['y'], dim['z']], dtype=np.float64)
						position = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64) 
						dim = [dimensions[2], dimensions[1], dimensions[0] ]
						rot = origin.get_rotation_matrix_from_xyz((0, 0, a['yaw']))
						center_cam, rot = convertTransform2Omni3D(origin, position, rot, dim)
						center = center_cam.flatten()

						omo = MapObjectTracker(category, center, dim, rot, 1.0, static=True)
						omo.uid = uid
						objs.append(omo)
						uid += 1

			elif roomName == "Room{}".format(roomNum):

				annos = rooms[r]['labels']['ground-truth']['attributes']['annotations']

				for a in annos:
					category = a['category_id']
					pos = a['position']
					dim = a['dimensions']
					dimensions = np.array([dim['x'], dim['y'], dim['z']], dtype=np.float64)
					position = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64) 
					dim = [dimensions[2], dimensions[1], dimensions[0] ]
					rot = origin.get_rotation_matrix_from_xyz((0, 0, a['yaw']))
					center_cam, rot = convertTransform2Omni3D(origin, position, rot, dim)
					center = center_cam.flatten()

					omo = MapObjectTracker(category, center, dim, rot, 1.0, static=True)
					omo.uid = uid
					objs.append(omo)
					uid += 1
	else:
		for r in range(len(rooms)):
			annos = rooms[r]['labels']['ground-truth']['attributes']['annotations']

			for a in annos:
				category = a['category_id']
				pos = a['position']
				dim = a['dimensions']
				dimensions = np.array([dim['x'], dim['y'], dim['z']], dtype=np.float64)
				position = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64) 
				dim = [dimensions[2], dimensions[1], dimensions[0] ]
				rot = origin.get_rotation_matrix_from_xyz((0, 0, a['yaw']))
				center_cam, rot = convertTransform2Omni3D(origin, position, rot, dim)
				center = center_cam.flatten()

				omo = MapObjectTracker(category, center, dim, rot, 1.0, static=True)
				omo.uid = uid
				objs.append(omo)
				uid += 1

	return objs