

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import numpy as np
from ObjectUtils import * 
from threading import Thread
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class MyThread(Thread):
	def __init__(self, callback):
	    Thread.__init__(self)
	    self.callback = callback
	    self.viz = True

	def run(self):
		while True:
			k = input()
			self.viz = False
			break

def my_callback(viz):
    viz = False


def plotIOUScatter(data, iou, title='', ptype='conf'):

	plt.title(title)
	plt.scatter(data, iou, s=1, marker='o', c='g')
	plt.xlabel(ptype)
	plt.ylabel('IOU')
	#plt.show()
	plt.savefig(ptype + '_' + title + ".png")
	plt.close('all')


def plotIOUHistogram(iou, dist, title=''):


	#dist_arr = np.asarray(dist)
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,8))
	fig.suptitle(title)

	counts1, bins1 = np.histogram(dist)
	counts2, bins2 = np.histogram(dist, weights=iou)
	ax1.hist(bins1[:-1], bins1, weights=counts1)
	ax2.hist(bins2[:-1], bins2, weights=counts2)
	ax3.hist(bins2[:-1], bins2, weights=counts2/counts1)


	#plt.show()
	plt.savefig("histogram_" + title + ".png")
	plt.close('all')



def plotIOUHeatMap(iou, pos, title=''):

	
	xmax = max(pos[:, 0])
	xmin = min(pos[:, 0])
	ymax = max(pos[:, 1])
	ymin = min(pos[:, 1])
	
	xedges = np.arange(xmin, xmax, step = 0.25)
	yedges = np.arange(ymin, ymax, step = 0.25)
	print(xedges, yedges)
	if len(xedges) <2:
		return
	if len(yedges) <2:
		return


	# print(xmin, xmax, xedges)
	# print(ymin, ymax, yedges)

	H, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=(xedges, yedges))
	H2, xedges2, yedges2 = np.histogram2d(pos[:, 0], pos[:, 1], bins=(xedges, yedges), weights=iou)

	H3 = H2 / H

	#if nx > ny:
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,8))
	#else:
	#	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,12))

	fig.suptitle(title)
	ax1.set_title('prediction poses')
	ax2.set_title('accumulated IOU')
	ax3.set_title('normalized IOU')
	ax1.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
	ax2.imshow(H2, extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
	ax3.imshow(H3, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

	#plt.show()
	plt.savefig(title + "_heatmap.png")
	plt.close('all')




def Debug3DText(mapObjs, detObjs=[], gt=None, all_classes=None):


	if len(mapObjs) == 0:
		print("no map objects")
		return


	gui.Application.instance.initialize()
	window = gui.Application.instance.create_window("Map Debugger", 1024, 750)
	clr = cm.rainbow(np.linspace(0, 1, 14))

	scene = gui.SceneWidget()
	scene.scene = rendering.Open3DScene(window.renderer)
	window.add_child(scene)

	origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
	scene.scene.add_geometry("origin", origin, rendering.MaterialRecord())

	pnts = np.empty((8,3))
	if gt:
		arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.3, cone_height=0.4,cylinder_radius=0.2,cylinder_height=1.5)
		arrow.rotate(origin.get_rotation_matrix_from_xyz((0,-gt[3],0)))
		arrow = arrow.translate((gt[0],gt[1],gt[2]))
		scene.scene.add_geometry("gt", arrow, rendering.MaterialRecord())


	for ind, o in enumerate(detObjs):

		xyz = getBoxFromObj(o)
		xyz = np.asarray(xyz).reshape((8, 3))
		#print(xyz.shape)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(xyz)
		#pnts = np.append(pnts, xyz, axis=0)

		bb_t = pcd.get_oriented_bounding_box()
		bb_t.color = clr[o.category][:3]
		scene.scene.add_geometry("bb_{}".format(ind), bb_t, rendering.MaterialRecord())
		l = scene.add_3d_label(o.center, "matched:{},conf:{:2f},skip:{}".format(o.times_matched, o.conf, o.times_skipped))
		l.color = gui.Color(r =clr[o.category][0], g = clr[o.category][1], b=clr[o.category][2], a=1.0)

	for ind, o in enumerate(mapObjs):

		xyz = getBoxFromObj(o)
		xyz = np.asarray(xyz).reshape((8, 3))
		#print(xyz.shape)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(xyz)
		#pnts = np.append(pnts, xyz, axis=0)

		bb_t = pcd.get_oriented_bounding_box()
		bb_t.color = [0, 0, 0]
		scene.scene.add_geometry("mm_{}".format(ind), bb_t, rendering.MaterialRecord())
		#scene.add_3d_label(o.center, "matched:{},skip:{}".format(o.times_matched, o.times_skipped))

	# print(pnts.shape)
	# pcd = o3d.geometry.PointCloud()
	# pcd.points = o3d.utility.Vector3dVector(pnts)
	#bb_all = pcd.get_oriented_bounding_box()

	xyz = np.array([[-5, -2, -12.5], [5, -2, -12.5], [-5, 2, 12.5], [5, 2, 12.5], [5, 2, -12.5]])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	bb_t = pcd.get_oriented_bounding_box()
	bounds = bb_t.get_axis_aligned_bounding_box()
	scene.setup_camera(60, bounds, bounds.get_center())
	th = MyThread(my_callback)
	th.start()
	while th.viz:
		gui.Application.instance.run_one_tick() 
		
	#print("done")
	gui.Application.instance.quit()
	th.join()
	

	return


def VisGTvsPredObjs(origin, gt_objs, pr_objs):

	cubes = []
	colors = []
	clr = cm.rainbow(np.linspace(0, 1, 14))

	for o in pr_objs:
		addObj2DebugViz(cubes, colors, o, clr[o.category][:3])

	for o in gt_objs:
		addObj2DebugViz(cubes, colors, o, [0, 0, 0])

	debug3DViz(origin, cubes, colors)



def addToDebugViz(cubes, colors, center, dim, rot, clr):

	box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
	verts, faces = get_cuboid_verts_faces(box3d, rot)
	xyz = np.asarray(verts).reshape((8, 3))
	cubes.append(xyz)
	colors.append(clr)


def addObj2DebugViz(cubes, colors, obj, clr):

	center = obj.center
	dim = obj.dim
	box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
	verts, faces = get_cuboid_verts_faces(box3d, obj.rot)
	xyz = np.asarray(verts).reshape((8, 3))
	cubes.append(xyz)
	colors.append(clr)

# visualizes all objects in the camera frame
def debug3DViz(origin, cubes, colors, args=None, im=None):

    bbs = []
    bbs.append(origin)

    for i, cube in enumerate(cubes):
        xyz = cube
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        bb_t = pcd.get_oriented_bounding_box()
        bb_t.color = colors[i]
        bbs.append(bb_t)

    # if im:
    #     bbs.append(im)
    if args:
        o3d.visualization.draw_geometries(bbs, zoom=args[0], front=args[1], lookat=args[2], up=args[3])
    else: 
        o3d.visualization.draw_geometries(bbs)


def visSeenObjects(origin, mapObjects, seenInd, detObjs):

	cubes = []
	colors = []

	clr = cm.rainbow(np.linspace(0, 1, 14))
	seenObjs = [mapObjects[i] for i in seenInd]
	unseenObjs = [mapObjects[i] for i in range(len(mapObjects)) if i not in seenInd]
	# print(len(mapObjects))
	# print(len(seenObjs))
	# print(len(unseenObjs))
	for o in seenObjs:
		if o.skip == removeTH - 1:
			addObj2DebugViz(cubes, colors, o, [1, 0, 0])
		else:
			addObj2DebugViz(cubes, colors, o, [0, 1, 0])

	for o in unseenObjs:
		if o.skip == removeTH - 1:
			addObj2DebugViz(cubes, colors, o, [1, 0, 0])
		else:
			addObj2DebugViz(cubes, colors, o, [1, 1, 0])

	for o in wallObjs:
		addObj2DebugViz(cubes, colors, o, [0, 0, 0])

	for o in detObjs:
		addObj2DebugViz(cubes, colors, o, [1, 0, 1])


	im = None
	# if imPath:
	# 	img = cv2.imread(imPath)
	# 	#im = o3d.geometry.Image(x)
		
	# 	cv2.imshow(imPath, img)
	# 	cv2.waitKey()
	# 	cv2.destroyAllWindows()


	zoom = 1.1412000000000007
	front = [ -0.10211633077463615, -0.98546095915029597, 0.13579010633954072]
	lookat = [ 2.6172, 2.0474999999999999, 1.532 ]
	up = [0.02048854291456121, 0.13439120603606852, 0.9907165201758823]
	args = [zoom, front, lookat, up]

	debug3DViz(origin, cubes, colors, args, im)

def visFinalObjects(origin, mapObjects):

	cubes = []
	colors = []

	clr = cm.rainbow(np.linspace(0, 1, 14))
	
	for o in mapObjects:
		addObj2DebugViz(cubes, colors, o, clr[o.category][:3])

	zoom = 1.1412000000000007
	front = [ -0.10211633077463615, -0.98546095915029597, 0.13579010633954072]
	lookat = [ 2.6172, 2.0474999999999999, 1.532 ]
	up = [0.02048854291456121, 0.13439120603606852, 0.9907165201758823]
	args = [zoom, front, lookat, up]

	debug3DViz(origin, cubes, colors, args)


def plotPairs(origin, gmap, matches_pairs, all_classes):

	gridmap = gmap.map.copy()
	gridmap = gridmap[:285, :]
	h, w = gridmap.shape
	gridmap = resizeMap(gridmap, 3)
	gridmap = cv2.cvtColor(gridmap,cv2.COLOR_GRAY2RGB)

	clr = cm.rainbow(np.linspace(0, 1, len(all_classes)))
	cubes = []
	colors = []

	for p in matches_pairs:
		pr_obj = p[0]
		gt_obj = p[1]
		iou = p[2]
		gt_pose = p[3]

		category = gt_obj.category

		xyz = getBoxFromObj(gt_obj)
		xyz = np.asarray(xyz).reshape((8, 3))
		cubes.append(xyz)
		colors.append(clr[category][:3])

		xyz = getBoxFromObj(pr_obj)
		xyz = np.asarray(xyz).reshape((8, 3))
		cubes.append(xyz)
		colors.append([0.5, 0.5, 0.5])

	debug3DViz(origin, cubes, colors)


def plotPred(origin, gmap, gt_objs, pr_objs, all_classes):

	gridmap = gmap.map.copy()
	gridmap = gridmap[:285, :]
	h, w = gridmap.shape
	gridmap = resizeMap(gridmap, 3)
	gridmap = cv2.cvtColor(gridmap,cv2.COLOR_GRAY2RGB)

	clr = cm.rainbow(np.linspace(0, 1, len(all_classes)))
	cubes = []
	colors = []

	for gt_obj in gt_objs:

		category = gt_obj.category
		xyz = getBoxFromObj(gt_obj)
		xyz = np.asarray(xyz).reshape((8, 3))
		cubes.append(xyz)
		colors.append(clr[category][:3])

	for pr_obj in pr_objs:

		category = pr_obj.category
		xyz = getBoxFromObj(pr_obj)
		xyz = np.asarray(xyz).reshape((8, 3))
		cubes.append(xyz)
		colors.append([0, 0, 0])
			

	debug3DViz(origin, cubes, colors)



def resizeMap(img, s):

	h, w = img.shape

	img2 = np.zeros((h*s, w*s), dtype="uint8")
	for x in range(w * s):
		for y in range(h * s):
			img2[y, x] = img[int(y/s), int(x/s)]

	return img2



# reads the json file that is produces by segements.ai and converts it into a 2D map of the floor with semantic information
def draw2DMap(gmap, objs, classes):

	scale = 1
	gridmap = gmap.map.copy()
	gridmap = gridmap[:285, :]
	h, w = gridmap.shape
	gridmap = resizeMap(gridmap, scale)
	gridmap = cv2.cvtColor(gridmap,cv2.COLOR_GRAY2RGB)

	semMaps = []
	for i in range(len(classes)):
		semImg = np.zeros((h, w), dtype=np.uint8)
		semMaps.append(semImg)

	clr = cm.rainbow(np.linspace(0, 1, len(classes)))
	for o in objs:

		pos = o.center 
		dim = o.dim
		category = o.category
		uid = o.uid
		box = obj2UVs(gmap, o, scale)
		color = 255 * clr[category][:3]
		cv2.drawContours(gridmap, [box], 0,  color, 2)
		cv2.putText(gridmap, "{}".format(uid), box[0] + [np.random.randint(10), 0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		box = obj2UVs(gmap, o, 1)
		#cv2.drawContours(semMaps[category], [box], 0,  uid+1, -1)
		cv2.drawContours(semMaps[category], [box], 0,  255, -1)

	# for i in range(len(classes)):
	# 	cv2.imwrite(classes[i] + ".png", semMaps[i])

	return gridmap, semMaps


# reads the json file that is produces by segements.ai and converts it into a 2D map of the floor with semantic information
def draw2DMap2(gmap, objs, classes):

    gridmap = gmap.map.copy()
    gridmap = gridmap[:285, :]
    h, w = gridmap.shape
    gridmap = resizeMap(gridmap, 3)
    gridmap = cv2.cvtColor(gridmap,cv2.COLOR_GRAY2RGB)

    semMaps = []
    for i in range(len(classes)):
        semImg = np.zeros((h, w), dtype=np.uint8)
        semMaps.append(semImg)

    # clr = cm.rainbow(np.linspace(0, 1, len(classes)))
    # for o in objs:

    #     pos = o.pos 
    #     dim = o.dim
    #     category = o.category
    #     gid = o.gid
    #     box = obj2UVs(gmap, o, 3)
    #     color = 255 * clr[category][:3]
    #     cv2.drawContours(gridmap, [box], 0,  color, 2)
    #     cv2.putText(gridmap, "{}".format(gid), box[0] + [np.random.randint(10), 0], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    #     box = obj2UVs(gmap, o, 1)
    #     cv2.drawContours(semMaps[category], [box], 0,  gid+1, -1)

    for i in range(len(classes)):
      cv2.imwrite(classes[i] + ".png", semMaps[i])

    return gridmap, semMaps



def Build2DMapFromMapObjects(gmap, mapObjects):	

	# gridmap = cv2.imread("/home/nickybones/Code/OmniNMCL/ncore/data/floor/JMap/JMap.png")
	# gmap = GMAP(gridmap, 0.05, [-13.9155, -24.59537, 0.0])
	origin = o3d.geometry.TriangleMesh.create_coordinate_frame()

	sem_maps = []
	for s in range(14):
		sem_maps.append(np.zeros((gmap.map.shape[:2]), np.int8))

	clr = cm.rainbow(np.linspace(0, 1, 14))

	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		center, dim, rot = toLabCoords(center, dim, rot)
		
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		for v in range(4):
			#vert = xyz_t[vertIDS[v]]
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		cv2.fillConvexPoly(sem_maps[category], box, 255)


	return sem_maps


def Build2DMapFromMapObjectsInLabFrame(gmap, mapObjects):	

	sem_maps = []
	for s in range(14):
		sem_maps.append(np.zeros((gmap.map.shape[:2]), np.uint8))

	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		if len(hull.vertices) < 4:
			continue

		for v in range(4):
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		cv2.fillConvexPoly(sem_maps[category], box, 255)

	return sem_maps

def Build2DBordersFromMapObjectsInLabFrame(gmap, mapObjects):	

	gridmap = copy.deepcopy(255 - gmap.map)
	gridmap = cv2.cvtColor(gridmap, cv2.COLOR_GRAY2RGB)
	clr = cm.rainbow(np.linspace(0, 1, 14))
	
	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		if len(hull.vertices) < 4:
			continue

		for v in range(4):
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		#cv2.fillConvexPoly(gridmap, box, 255 * clr[category], cv2.LINE_4)
		cv2.drawContours(gridmap, [box], 0,  255 * clr[category], 2)

	return gridmap


def Build2DBordersFromMapObjects(gmap, mapObjects):	

	gridmap = copy.deepcopy(255 - gmap.map)
	gridmap = cv2.cvtColor(gridmap, cv2.COLOR_GRAY2RGB)
	clr = cm.rainbow(np.linspace(0, 1, 14))
	
	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf
		center, dim, rot = toLabCoords(center, dim, rot)
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		if len(hull.vertices) < 4:
			continue

		for v in range(4):
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		#cv2.fillConvexPoly(gridmap, box, 255 * clr[category], cv2.LINE_4)
		cv2.drawContours(gridmap, [box], 0,  255 * clr[category], 2)

	return gridmap



def Build2DInstacnceMapsFromMapObjects(gmap, mapObjects):	


	sem_maps = []
	for s in range(14):
		sem_maps.append(np.zeros((gmap.map.shape[:2]), np.int32))

	cnt = 1
	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		center, dim, rot = toLabCoords(center, dim, rot)
		
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		for v in range(4):
			#vert = xyz_t[vertIDS[v]]
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		cv2.fillConvexPoly(sem_maps[category], box, cnt)
		cnt += 1


	return sem_maps