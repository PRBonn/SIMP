#!/usr/bin/env python3

import cv2
import pandas as pd
import time
import open3d as o3d
from MapObjectTracker import MapObjectTracker
from scipy.spatial.transform import Rotation as R
import json
from GMAP import GMAP 
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib import cm
from pytorch3d.ops import box3d_overlap
from VisUtils import *
from JSONUtils import *
from ObjectUtils import *
from DatasetUtils import *
import copy
from scipy.spatial import distance

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


def angdist(ang1, ang2):

	dist = distance.cosine([np.cos(ang1), np.sin(ang1)], [np.cos(ang2), np.sin(ang2)])
	return np.abs(dist * np.pi * 0.5)



def Build2DInstacnceMapsFromMapObjects(gmap, mapObjects):	

	clr = cm.rainbow(np.linspace(0, 1, 14))

	sem_maps = []
	for s in range(14):
		sem_maps.append(np.zeros((gmap.map.shape[:2]), np.int32))

	gridmap = cv2.cvtColor(gmap.map, cv2.COLOR_GRAY2RGB)

	cnt = 1
	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		#center, dim, rot = toLabCoords(center, dim, rot)
		
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz_t = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz_t[:, :2])
		box = np.zeros((4, 2), np.int32)

		if len(hull.vertices) < 4:
			continue

		for v in range(4):
			#vert = xyz_t[vertIDS[v]]
			vert = xyz_t[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv


		cv2.fillConvexPoly(sem_maps[category], box, cnt)
		cnt += 1
		cv2.fillConvexPoly(gridmap, box, 255*clr[category][:3])


	return sem_maps, gridmap


def Build2DBinaryMapsFromMapObjects(gmap, mapObjects):	


	sem_maps = []
	for s in range(14):
		sem_maps.append(np.zeros((gmap.map.shape[:2]), np.uint8))

	for o in mapObjects:

		center = o.center
		dim = o.dim
		rot = o.rot
		category = o.category
		conf = o.conf

		#center, dim, rot = toLabCoords(center, dim, rot)
		
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

		# cv2.imshow("", sem_maps[mapObjects[0].category])
		# cv2.waitKey()
		

	return sem_maps


def ggd(data, mx, my, sx, sy, b):
		#return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**b / (2. * sx**b) + (data[:, 1] - my)**b / (2. * sy**b)))
		return  np.exp(-((data[0] - mx)**b / (2. * sx**b) + (data[1] - my)**b / (2. * sy**b)))


class SIMP():

	def __init__(self, jsonPath):

		with open(jsonPath) as f:
			self.args = json.load(f)

		self.mapObjects = []
		self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
		self.gmap = GMAP(self.args['gmap'])
		self.categories = self.args['categories']
		self.semMaps, self.map2d = Build2DInstacnceMapsFromMapObjects(self.gmap, self.mapObjects)
		self.prev_pose = np.array([0, 0, 0])
		self.clr = cm.rainbow(np.linspace(0, 1, len(self.categories)))
		self.df = pd.read_csv(self.args['variancePath'])
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.img_h = 480
		self.img_w = 640
		self.debug_render = None
		self.roomseg = cv2.imread(self.args['roomseg'], 0)
		self.curr_room = 0
		self.detc_cnt = 0
		self.detObjs = []
		self.forbidden_relations = [(0, 4), (4, 0), (4, 9), (9, 4), (8, 7), (7, 8), (8, 9), (9, 8), (13, 4), (4, 13), (4, 8), (8, 4), (2, 7), (7, 2), (6, 9), (9, 6)]
		self.K = np.reshape(np.array([self.args['K']]), (3,3))
		self.cam_poses = np.reshape(np.array([self.args['cam_poses']]), (4,4))
		self.angles = self.args['angles']

	def sampleDist(self, detObj, mapObj):

		if detObj.category != mapObj.category:
			return 0.0

		category = detObj.category
		row = self.df[self.df['category'] == category]
		mx = row['mx'].to_numpy()[0]
		my = row['my'].to_numpy()[0]
		sx = row['sx'].to_numpy()[0]
		sy = row['sy'].to_numpy()[0]
		b = row['b'].to_numpy()[0]

		detUV = self.gmap.world2map(detObj.center)
		center_2d = self.gmap.world2map(mapObj.center)

		r = R.from_matrix(mapObj.rot)
		roll, pitch, yaw = r.as_euler('xyz', degrees=False)
		R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
		mx_ = mx + center_2d[0]
		my_ = my + center_2d[1]
		s = np.array([[sx, 0.0], [0.0, sy]])
		s_ = R2d @ s @ R2d.T
		sx_ = s_[0][0]
		sy_ = s_[1][1]

		p = ggd(detUV, mx_, my_, sx_, sy_, b)
		#print(p)
		return p



	def update(self, detections, gt):

		# check odometry trigger
		curr_pose = np.array(gt)
		# if self.args['odomTrigger']:
		# 	if np.linalg.norm(curr_pose[:2] - self.prev_pose[:2]) < self.args['odom_trigerTH'][0]:
		# 		return


		
		if np.linalg.norm(curr_pose[:2] - self.prev_pose[:2]) < self.args['odom_trigerTH'][0] and angdist(curr_pose[3], self.prev_pose[3]) < self.args['odom_trigerTH'][1]:

			self.detc_cnt += 1

			curr_detc = []
			for d in detections:
			
				category = d.category
				conf = d.confidence
				gt_ = copy.deepcopy(gt)

				# if conf < self.args['omni_conf'][category]:
				# 	continue
				gt_[2] = 0
				center, dim, rot = self.moveToGlobal(gt_, d.center, d.dim, np.reshape(d.rot, (3,3)))

				uv = self.gmap.world2map([center[0], center[1]])
				if uv[0] >= 0 and uv[1] >= 0 and uv[0] < 521 and uv[1] < 280:
					room = self.roomseg[uv[1], uv[0]]
					#print("object room is {}".format(room))
					if room != self.curr_room:
						continue
				else:
					continue

				#print(gt, center, dim, rot, conf, category)
				obj = MapObjectTracker(category, center, dim, rot, conf,room)

				# if self.args['2DvisTest']:
				# 	vis = isVisible(self.gmap, [gt[0], gt[1]], [obj.center[0], obj.center[1]])
				# 	if not vis:
				# 		continue

				self.detObjs.append(obj)
				
			return

		else:

			self.prev_pose = curr_pose
			gt_uv = self.gmap.world2map([gt[0], gt[1]])
			self.curr_room = self.roomseg[gt_uv[1], gt_uv[0]]
			#print("current room is {}".format(self.curr_room))

		if len(self.detObjs):

			#print("detected object before merge {}".format(len(self.detObjs)))
			start = time.time()
			self.detObjs = self.mergeObjects(self.detObjs, 0.7)
			self.detObjs = self.purgeDetections(self.detObjs)
			end = time.time()
			#print("merge detection time ", end - start)
			self.detc_cnt = 0
			#print("detected object after merge {}".format(len(self.detObjs)))

			# check seen objects 
			start = time.time()
			self.getSeenObjects(gt)
			end = time.time()
			#print("render time ", end - start)

			activeObjs = []
			for o in self.mapObjects:
				if o.active > 0:
					activeObjs.append(o)

			for o in self.detObjs:
				category = o.category
				clsObjs = [s for s in self.mapObjects if s.category == category]
				if len(clsObjs) == 0:
					self.mapObjects.append(o)
			if len(activeObjs) == 0:
				for o in self.detObjs:
					self.mapObjects.append(o)

			else:
				cost = self.computeCost(self.detObjs, activeObjs)
				# hungrian algo
				row_ind, col_ind = linear_sum_assignment(cost)

				for rind in range(len(row_ind)):
					obj_cost = cost[row_ind[rind], col_ind[rind]]
					# if some cost is too big, create a new instance
					th = 1.0 - self.args['costTH']
					det = self.detObjs[row_ind[rind]]
					if obj_cost > th:
						self.mapObjects.append(det)
					else:
						#activeObjs[col_ind[rind]].update(det.center, det.dim, det.rot)
						activeObjs[col_ind[rind]].mergeObject(det)

				notSelected = [oid for oid in range(len(self.detObjs)) if oid not in row_ind]
				
				for ns in notSelected:
					det = self.detObjs[ns]
					self.mapObjects.append(det)


			self.semMaps, self.map2d = Build2DInstacnceMapsFromMapObjects(self.gmap, self.mapObjects)
			self.detObjs.clear()

		n = len(self.mapObjects)
		start = time.time()
		self.mapObjects = self.mergeObjects(self.mapObjects, 0.7)
		self.purge()
		self.purgRelation()
		m = len(self.mapObjects)
		end = time.time()
		#print("merge objects time ", end - start)
		
		



	def moveToGlobal(self, gt, center, dim, rot):

		T = np.eye(4)
		R = self.origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
		T[:3, :3] = R
		T[0, 3] = gt[0]
		T[1, 3] = gt[1]
		T[2, 3] = gt[2]

		center = np.array([center[0], center[1], center[2], 1.0]).T
		rot = np.array(rot)

		center = (T @ center).T.flatten()
		center = center[:3]
		rot = R @ rot

		return center, dim, rot


	def computeCost(self, detObjs, clsObjs):

		n = len(clsObjs)
		m = len(detObjs)

		cost = np.zeros((m, n), dtype=np.float64)
		for i in range(m):
			for j in range(n):
				
				boxes1 = getBoxFromObj(detObjs[i])
				boxes2 = getBoxFromObj(clsObjs[j])
				intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
				iou_3d = iou_3d.cpu().detach().numpy().item()
				#tmp = (1.0 - iou_3d) + 1000 * (1 - (detObjs[i].category == clsObjs[j].category)) 
				cost[i][j] = (1.0 - iou_3d) + 1000 * (1 - (detObjs[i].category == clsObjs[j].category)) + (1.0 - self.sampleDist(detObjs[i], clsObjs[j]))
				cost[i][j] *= 0.5

		return cost





	def mergeObjectsSingle(self, objs, TH):

		newObjs = []
		roomObjs = [obj for obj in objs if obj.room == self.curr_room]

		for cat in [0,2, 3,4, 6,7, 8,9,11,13]:
			semMapObjs = FindAllObjectsByCategoty(roomObjs, cat)
			semMapObjs = self.centerMerge(semMapObjs)

			for i in range(len(semMapObjs)):
				o1 = semMapObjs[i]
				#print(o1.skip)
				semNewObjs = FindAllObjectsByCategoty(newObjs, cat)
				added = False
				for j in range(len(semNewObjs)):
					o2 = semNewObjs[j]
					boxes1 = getBoxFromObj(o1)
					boxes2 = getBoxFromObj(o2)
					intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
					cost = 1.0 - iou_3d.cpu().detach().numpy().item()
					cost += (1.0 - self.sampleDist(o1, o2))
					cost *= 0.5
					dist = np.linalg.norm(o1.center[:2] - o2.center[:2])
					# consider center distance
					if cost < TH and o1.room == o2.room and dist < 1.0:
						o2.mergeObject(o1)
						added = True
					
				if not added:
					newObjs.append(o1)

		#self.mapObjects = newObjs
		newObjs.extend([obj for obj in objs if obj.room != self.curr_room])

		return newObjs

	def centerMerge(self, objs):
		newObjs = []
		
		for o1 in objs:
			added = False
			for o2 in newObjs:
				if o1.uid == o2.uid or o1.room != o2.room:
					continue


				centerTH = 0.5
				if o2.category == 8:
					centerTH = 1.0

				if np.linalg.norm(o1.center[:2] - o2.center[:2]) < centerTH:
					o2.mergeObject(o1)
					added = True
					
			if not added:
				newObjs.append(o1)

		return newObjs



	def mergeObjects(self, objs, TH):

		n = len(objs)
		m = 0
		while (m != n):
			n = len(objs)
			#print("before {}", n)
			objs = self.mergeObjectsSingle(objs, TH)
			m = len(objs)
			#print("after {}", m)

		return objs



	def get2dintersection(self, o1, o2):

		sem_maps = Build2DBinaryMapsFromMapObjects(self.gmap, [o1, o2])
		sem_maps[o1.category] = sem_maps[o1.category]/ 255
		sem_maps[o2.category] = sem_maps[o2.category]/ 255
		a1 = np.count_nonzero(sem_maps[o1.category])
		a2 = np.count_nonzero(sem_maps[o2.category])
		inter_img = sem_maps[o2.category] * sem_maps[o1.category]
		intersection = np.count_nonzero(inter_img)

		return a1, a2, intersection


	def purgRelation(self):

		removeInd = []

		for i, o1 in enumerate(self.mapObjects):
			if o1.room != self.curr_room:
				continue
			for j, o2 in enumerate(self.mapObjects):
				if o2.uid == o1.uid or o2.category == o1.category:
					continue
				if (o1.category, o2.category) in self.forbidden_relations:
					boxes1 = getBoxFromObj(o1)
					boxes2 = getBoxFromObj(o2)
					# intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
					# v1 = intersection_vol/ o1.volume
					# v2 =  intersection_vol/ o2.volume
					# forbidden_iou = np.max([v1, v2])
					a1, a2, intersection = self.get2dintersection(o1, o2)
					v1 = intersection/ a1
					v2 =  intersection/ a2
					forbidden_iou = np.max([v1, v2])

					if forbidden_iou > 0.5:

						if o1.times_matched > o2.times_matched:
							removeInd.append(j)
						elif o1.times_matched < o2.times_matched:
							removeInd.append(i)

		diet = [self.mapObjects[i] for i in range(len(self.mapObjects)) if i not in removeInd]
		self.mapObjects = diet

	def purge(self):

		removeInd = []
		for oInd, o in enumerate(self.mapObjects):
			#print(o.skip)
			if o.room == self.curr_room:
				if o.active or o.inactive > 100:
					if o.skip >= self.args['skipTH']:
						if float(o.times_matched) / o.skip < 0.5 :   
							removeInd.append(oInd)
				#if o.skip >= self.args['skipTH']:
					#removeInd.append(oInd)
					#print(o.category)

		diet = [self.mapObjects[i] for i in range(len(self.mapObjects)) if i not in removeInd]
		self.mapObjects = diet
		#print("after remove {}", len(mapObjects))

	def purgeDetections(self, objs):

		removeInd = []
		for oInd, o in enumerate(objs):
			if o.times_matched < self.detc_cnt * 0.3:
				removeInd.append(oInd)
				

		diet = [objs[i] for i in range(len(objs)) if i not in removeInd]
		objs = diet

		return objs
		#print("after remove {}", len(mapObjects))




	def getSeenObjects(self, gt):

		visObjs = []
		for i, o in enumerate(self.mapObjects):
			vis = isVisibleSem(self.gmap, self.semMaps[o.category], i + 1, [gt[0], gt[1]], [o.center[0], o.center[1]])
			if vis and o.room == self.curr_room:
				visObjs.append(o)
				o.predict(True)
			else:
				o.predict(False)

		if len(visObjs) == 0:
			return

		im_rendered, fragment, meshes, clrs, truncations = self.renderMeshes2(self.origin, visObjs, gt, self.device)
		if im_rendered is None:
			return
		
		for i, o in enumerate(visObjs):

			clr = list(0.5 * clrs[i][:3])
			# set red range
			lowcolor = (clr[0]-0.01,clr[1]-0.01,clr[2]-0.01)
			highcolor = (clr[0]+0.01,clr[1]+0.01,clr[2]+0.01)

			# threshold
			thresh = cv2.inRange(im_rendered, lowcolor, highcolor)
			count = np.sum(thresh[np.nonzero(thresh)]) 
			#print(count / (self.img_h * self.img_w))
			# cv2.imshow("", im_rendered)
			# cv2.waitKey()
			#if count :
			if float(count) / (255.0 * self.img_h * self.img_w) > 0.01:
				#print(float(count) / (255.0 * self.img_h * self.img_w))
				trunc = truncations[i]
				if trunc > self.args['truncTH']:
					o.predict(False)
					continue
				#print(o.category)
				o.predict(True)

			else:
				o.predict(False)




	def gt2Omni3DCoords2(self, origin, gt):

		R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
		R_inv = np.linalg.inv(R)
		# R_gt = origin.get_rotation_matrix_from_xyz((0, 0, gt[3]))
		# R_gt = R_inv @ R_gt

		center = np.ones((1, 3), dtype=np.float64)
		center[:, :] = gt[:3]
		center = center.T
		center_cam = R_inv @ center
		gt[0] = center_cam[0].item()
		gt[1] = center_cam[1].item()
		gt[2] = center_cam[2].item()

		return gt

	def convertTransform2Omni3D2(self, origin, center_cam, R_cam, dimensions):

		R = origin.get_rotation_matrix_from_xyz((-0.5 * np.pi, 0.5 * np.pi, 0))
		R_inv = np.linalg.inv(R)
		R_cam = R_inv @ R_cam
		center = np.ones((1, 3), dtype=np.float64)
		center[:, :] = center_cam
		center = center.T
		center_cam = R_inv @ center

		return center_cam, R_cam


	def renderMeshes2(self, origin, mapObjs, gt, device):

		clrs = cm.rainbow(np.linspace(0, 1, len(mapObjs)))
		meshes = []

		cubes = []
		colors = []
		truncs = []

		gt_ = copy.deepcopy(gt)
		gt_ = self.gt2Omni3DCoords2(origin, gt_)

		for i, o in enumerate(mapObjs):
			center = copy.deepcopy(o.center)
			rot = copy.deepcopy(o.rot)
			dim = copy.deepcopy(o.dim)
			clr = list(clrs[i][:3])

			# move to camer/omni frame
			center, rot = self.convertTransform2Omni3D2(origin, center, rot, dim)
			#print(o.category, center)
			
			# # get relative pose to camera
			rot, center = getComposedTransformOmni(origin, gt_, [center[0], center[1], center[2], 1], rot)
			center = center.flatten()

			# create mesh 
			box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
			mesh = mesh_cuboid(box3d, rot, color=clr)
			meshes.append(mesh)

			truncation = estimate_truncation(self.K, box3d, rot, self.img_w, self.img_h)
			truncs.append(truncation)

		cam_num = 1
		R = np.zeros((cam_num, 3, 3))
		T = np.zeros((cam_num, 3))
		im_rendered = None
		fragment = None
		meshes_scenes = []

		for c in range(cam_num):

			trans = copy.deepcopy(self.cam_poses[c])
			trans = self.gt2Omni3DCoords2(origin, trans)
			trans = trans[:3]
			R[c, :, :] =  np.expand_dims(origin.get_rotation_matrix_from_xyz((0, -self.angles[c], 0)), axis=0)
			T[c, :] = np.expand_dims(trans, axis=0)
			
		if meshes:	
			
			cameras = get_camera(self.K, self.img_w, self.img_h, R=R, T=T).to(device)
			renderer = get_basic_renderer(cameras, self.img_w, self.img_h, use_color=True).to(device)
			meshes_scene = join_meshes_as_scene(meshes).cuda()
			meshes_scene.textures = meshes_scene.textures.to(device)
			meshes_scenes = meshes_scene.extend(cam_num)
			#print(len(meshes_scenes))

			im_rendered, fragment = renderer(meshes_scenes)
			im_rendered = im_rendered.cpu().numpy()
			im = np.zeros((480, 640 * cam_num, 3), dtype=np.float64)
			for c in range(cam_num):
				im[:, c*640:(c+1)*640, :] = im_rendered[c, :, :, :3]

			self.debug_render = im
			im_rendered = im
			
		return im_rendered, fragment, meshes, clrs, truncs




	def renderSingleMeshes2(self, origin, obj, gt, device):

		center = obj.center
		rot = obj.rot
		dim = obj.dim

		# get relative pose to camera
		rot, center = getComposedTransformOmni(origin, gt, [center[0], center[1], center[2], 1], rot)

		# convert to omni coordinates
		center = center.flatten()

		# create mesh 
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		mesh = mesh_cuboid(box3d, rot)

		cameras = get_camera(self.K, self.img_w, self.img_h).to(device)
		renderer = get_basic_renderer(cameras, self.img_w, self.img_h).to(device)
		mesh = mesh.to(device)

		im_rendered, fragment = renderer(mesh)

		return im_rendered, fragment

	def getDepthMap(self, fragment):

		zbuf = fragment.zbuf[:, :, :, 0]
		zbuf[zbuf==-1] = math.inf
		depth_map, depth_map_inds = zbuf.min(dim=0)

		return depth_map



