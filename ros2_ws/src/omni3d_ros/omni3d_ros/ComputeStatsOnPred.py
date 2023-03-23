import json 
import os
import numpy as np
from DatasetUtils import *
from pytorch3d.ops import box3d_overlap
from GMAP import GMAP
import cv2
from ObjectUtils import *
from JSONUtils import *
from VisUtils import *
from scipy.optimize import curve_fit
from scipy.special import gamma, factorial
from tqdm import * 
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R

all_classes = ['sink', 'door', 'oven', 'board', 'table', 'box', 'potted plant', 'drawers', 'sofa', 'cabinet', 'chair', 'fire extinguisher', 'person', 'desk']

origin = o3d.geometry.TriangleMesh.create_coordinate_frame()




def JSONObjects2MapObject(o):

	mobj = MapObjectTracker(o['category_id'], o['center_cam'], o['dimensions'], np.reshape(o['R_cam'], (3,3)), o['score'])

	return mobj



def moveToGlobal(origin, gt, center, dim, rot):


	T = np.eye(4)
	R = origin.get_rotation_matrix_from_xyz((0, -gt[3], 0))
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


def MatchGT2Pred(pr_obj, gt_objs, gt, TH=np.inf):

	# center, dim, rot = moveToGlobal(origin, gt, pr_obj.center, pr_obj.dim, pr_obj.rot)
	# box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
	# verts, faces = get_cuboid_verts_faces(box3d, rot)
	# prBox = torch.unsqueeze(verts, dim=0)
	prBox = getBoxFromObj(pr_obj)

	best_iou = -1
	#best_iou = 0
	best_id = -1
	mindist = 1000000000

	
	for i in range(len(gt_objs)):

		#print("gt: ",  gt_objs[i].category)
		if pr_obj.category == gt_objs[i].category:
			gtBox = getBoxFromObj(gt_objs[i])
			intersection_vol, iou_3d = box3d_overlap(gtBox, prBox)
			iou_3d = iou_3d.cpu().detach().numpy().item()
			#print(iou_3d)
			if iou_3d > best_iou:
				best_iou = iou_3d
				best_id = i
			if best_iou == 0:
				dist = np.linalg.norm(gt_objs[i].center - pr_obj.center)
				if dist > TH:
					continue
				if dist < mindist:
					mindist = dist
					best_id = i

	

	return best_id, best_iou



def plotPairs(gmap, matches_pairs):

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

		if True:
			xyz = getBoxFromObj(gt_obj)
			xyz = np.asarray(xyz).reshape((8, 3))
			cubes.append(xyz)
			colors.append(clr[category][:3])

			#if pr_obj.conf > 0.8 and iou > 0.5:

			#if iou > 0.2:
			if True:

				center, dim, rot = moveToGlobal(origin, gt_pose, pr_obj.center, pr_obj.dim, pr_obj.rot)
				box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
				verts, faces = get_cuboid_verts_faces(box3d, rot)
				xyz = torch.unsqueeze(verts, dim=0)
				xyz = np.asarray(xyz).reshape((8, 3))
				cubes.append(xyz)
				colors.append([0.5, 0.5, 0.5])

	debug3DViz(origin, cubes, colors)


def findBestParams(xdata, ydata):

	min_res = 10000000
	best_mx = None
	best_my = None
	best_sx = None
	best_sy = None
	best_b = None
	best_fit = None

	for b in [2, 4, 6, 8]:
	#for b in [2]:

		def gaus2dfancy(data, mx, my, sx, sy):
			return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**b / (2. * sx**b) + (data[:, 1] - my)**b / (2. * sy**b)))

		parameters, covariance = curve_fit(gaus2dfancy, xdata, ydata)
		mx = parameters[0]
		my = parameters[1]
		sx = parameters[2]
		sy = parameters[3]

		#b = parameters[4]
		fit_y = gaus2dfancy(xdata, mx, my, sx, sy)
		residual = np.linalg.norm(fit_y - ydata)
		print(b, parameters, residual)
		if residual < min_res:
			min_res = residual
			best_mx = mx
			best_my = my
			best_sx = sx
			best_sy = sy
			best_b = b
			best_fit = fit_y


	return min_res, best_mx, best_my, best_sx, best_sy, best_b, best_fit


def findBestParams4center(xdata, ydata):

	min_res = 10000000
	best_mx = None
	best_my = None
	best_sx = None
	best_sy = None
	best_b = None
	best_fit = None

	def gauss(data, m, s):
		return 1.0 / np.sqrt((2. * np.pi * s**2)) * np.exp(-((data - m)**2 / (2. * s**2)))

	parameters, covariance = curve_fit(gauss, xdata, ydata)
	mx = parameters[0]
	sx = parameters[1]
	
	fit_y = gauss(xdata, mx, sx)
	residual = np.linalg.norm(fit_y - ydata)
	print(parameters, residual)
	
	return residual, mx, 0, sx, 0, 2, fit_y



def estimateVarinacePerObject(gmap, gt_objs):

	category = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05


	for gt_obj in gt_objs:

		df = pd.read_pickle("stats/images/heatmap2_" + str(gt_obj.uid) + ".pickle")
		h = df['h'][0]
		w = df['w'][0]
		im = df['data']
		im[im <= 10] = 0

		center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
		#box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		#verts, faces = get_cuboid_verts_faces(box3d, rot)
		box3d = [0, 0, 0, dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, origin.get_rotation_matrix_from_xyz((0, 0, 0)))
		verts = np.asarray(verts).reshape((8, 3))
		xyz = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz[:, :2])
		box = np.zeros((4, 2), np.int32)
		

		for v in range(4):
			vert = xyz[hull.vertices[v]]
			#uv = gmap.world2map(vert)
			box[v] = [int((10 + vert[0]) / resolution),int((10 + vert[1]) / resolution)]
		 	#box[v] = uv

		avg = np.mean(box, axis=0)
		# avg = np.array([0, 0])

		# def generalGaus2d(data, mx, my,bx, by, sx, sy):
		# 	return bx * by / (2. * np.pi * sx * sy * gamma(1.0/bx) * gamma(1.0/by) ) * np.exp(-((data[:, 0] - mx)**bx / (2. * sx**bx) + (data[:, 1] - my)**by / (2. * sy**by)))

		# def gaus2d(data, mx, my, sx, sy):
		# 	return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**2. / (2. * sx**2.) + (data[:, 1] - my)**2. / (2. * sy**2.)))

		# b = 4
		# def gaus2dfancy(data, mx, my, sx, sy):
		# 	return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**b / (2. * sx**b) + (data[:, 1] - my)**b / (2. * sy**b)))


		x = range(w) - avg[0]
		y = range(h) - avg[1]
		xv, yv = np.meshgrid(x, y)
		xv = xv.flatten()
		yv = yv.flatten()
		xdata = np.vstack((xv, yv)).T
		ydata = np.expand_dims(im, axis=1)
		nz_ids = np.nonzero(ydata)[0]
		xdata = xdata[nz_ids]
		ydata = ydata[nz_ids].squeeze()
		ydata = ydata / sum(ydata)
		if len(np.unique(ydata)) < 20:
			category.append(gt_obj.category)
			uid.append(gt_obj.uid)
			mxs.append(None)
			mys.append(None)
			sxs.append(None)
			sys.append(None)
			bs.append(None)
			corners.append(box.flatten())
			continue


		res, mx, my, sx, sy, b, fit_y = findBestParams(xdata, ydata)

		category.append(gt_obj.category)
		uid.append(gt_obj.uid)
		mxs.append(mx)
		mys.append(my)
		sxs.append(sx)
		sys.append(sy)
		bs.append(b)
		corners.append(box.tolist())


		# from mpl_toolkits import mplot3d
		# fig = plt.figure(figsize=plt.figaspect(0.5))
		# fig.suptitle(str(gt_obj.uid))
		# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		# ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata, c=ydata, label='data')
		# ax2.scatter3D(xdata[:, 0], xdata[:, 1], fit_y, c=fit_y, label='fit');
		# ax1.legend()
		# ax2.legend()
		# plt.show()
		# plt.close()

	datadic = {'category': category, 'uid': uid, 'mxs' : mxs, 'mys': mys, 'sxs': sxs, 'sys' : sys, 'bs' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/per_object_variance.csv', index=False)


def plotGlobalVariane(gmap, gt_objs):

	df = pd.read_csv('stats/variance.csv')
	resolution = 0.05

	def gaus2dfancy(data, mx, my, sx, sy, b):
			return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**b / (2. * sx**b) + (data[:, 1] - my)**b / (2. * sy**b)))



	x = range(521)
	y = range(280)
	xv, yv = np.meshgrid(x, y)
	xv = xv.flatten()
	yv = yv.flatten()
	xdata = np.vstack((xv, yv)).T


	fig = plt.figure(figsize=plt.figaspect(0.5))

	cnt = 1
	for category in [0, 2, 3, 4, 6, 7, 8, 9, 11, 13]:
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((280 * 521))

		b = df_cat['b'].to_numpy()
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:

			center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			r = R.from_matrix(rot)
			roll, pitch, yaw = r.as_euler('xyz', degrees=False)
			R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]
			ydata += gaus2dfancy(xdata, mx_, my_, sx_, sy_, b)
			#print(gt_obj.uid, center_2d)

		
		ax1 = fig.add_subplot(2, 5, cnt, projection='3d')
		ax1.set_title(all_classes[category])
		ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata, c=ydata, label='fit')
		ax1.invert_yaxis()
		ax1.legend()
		cnt += 1

	plt.show()
	plt.close()
			
	

def estimateGlobalVariance(gmap, gt_objs):

	categories = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sxys = []
	syxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05

	df = pd.read_csv('stats/variance.csv')


	for category in range(14):
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((280 * 521))

		b = df_cat['b'].to_numpy()[0]
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:

			center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			r = R.from_matrix(rot)
			roll, pitch, yaw = r.as_euler('xyz', degrees=False)
			R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]
			sxy = s[0][1]
			syx = s[1][0]

			box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
			verts, faces = get_cuboid_verts_faces(box3d, rot)
			verts = np.asarray(verts).reshape((8, 3))
			xyz = verts[verts[:, 2].argsort()][:4]
			hull = ConvexHull(xyz[:, :2])
			box = np.zeros((4, 2), np.int32)
		

			for v in range(4):
				vert = xyz[hull.vertices[v]]
				uv = gmap.world2map(vert)
				box[v] = uv

			categories.append(gt_obj.category)
			uid.append(gt_obj.uid)
			mxs.append(mx_)
			mys.append(my_)
			sxs.append(sx_)
			sys.append(sy_)
			syxs.append(syx)
			sxys.append(sxy)
			bs.append(b)
			corners.append(box.flatten().tolist())	

	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'syx': syxs, 'sxy' : sxys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/global_variance.csv', index=False)
	print(corners)

	import json 
	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys,  'syx': syxs, 'sxy' : sxys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)

	with open('stats/global_variance.json', 'w', encoding='utf-8') as f:
		json.dump(datadic, f, indent=4)



def estimateGlobalVarianceInLabFrame(gmap, gt_objs):

	categories = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sxys = []
	syxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05

	df = pd.read_csv('stats/variance.csv')


	for category in range(14):
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((280 * 521))

		b = df_cat['b'].to_numpy()[0]
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:

			center, dim, rot = gt_obj.center, gt_obj.dim, gt_obj.rot
			#center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			r = R.from_matrix(rot)
			roll, pitch, yaw = r.as_euler('xyz', degrees=False)
			R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]
			sxy = s[0][1]
			syx = s[1][0]

			box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
			verts, faces = get_cuboid_verts_faces(box3d, rot)
			verts = np.asarray(verts).reshape((8, 3))
			xyz = verts[verts[:, 2].argsort()][:4]
			hull = ConvexHull(xyz[:, :2])
			box = np.zeros((4, 2), np.int32)
		

			for v in range(4):
				vert = xyz[hull.vertices[v]]
				uv = gmap.world2map(vert)
				box[v] = uv

			categories.append(gt_obj.category)
			uid.append(gt_obj.uid)
			mxs.append(mx_)
			mys.append(my_)
			sxs.append(sx_)
			sys.append(sy_)
			syxs.append(syx)
			sxys.append(sxy)
			bs.append(b)
			corners.append(box.flatten().tolist())	

	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'syx': syxs, 'sxy' : sxys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/global_variance.csv', index=False)
	print(corners)

	import json 
	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys,  'syx': syxs, 'sxy' : sxys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)

	with open('stats/global_variance.json', 'w', encoding='utf-8') as f:
		json.dump(datadic, f, indent=4)



def plotGlobalVarianeInLabFrame(gmap, gt_objs):

	df = pd.read_csv('stats/variance.csv')
	resolution = 0.05

	def gaus2dfancy(data, mx, my, sx, sy, b):
			return 1.0 / (2. * np.pi * sx * sy) * np.exp(-((data[:, 0] - mx)**b / (2. * sx**b) + (data[:, 1] - my)**b / (2. * sy**b)))


	x = range(521)
	y = range(280)
	xv, yv = np.meshgrid(x, y)
	xv = xv.flatten()
	yv = yv.flatten()
	xdata = np.vstack((xv, yv)).T


	fig = plt.figure(figsize=plt.figaspect(0.5))

	cnt = 1
	#for category in [0, 2, 3, 4, 6, 7, 8, 9, 11, 13]:
	for category in [0]:
		gridmap = 255 - copy.deepcopy(gmap.map)
		gridmap = gridmap.T
		gridmap = cv2.cvtColor(gridmap,cv2.COLOR_GRAY2RGB)
	
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((280 * 521))

		b = df_cat['b'].to_numpy()
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:
			center, dim, rot = gt_obj.center, gt_obj.dim, gt_obj.rot
			#center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			r = R.from_matrix(rot)
			roll, pitch, yaw = r.as_euler('xyz', degrees=False)
			R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]
			ydata += gaus2dfancy(xdata, mx_, my_, sx_, sy_, b)
			#print(gt_obj.uid, center_2d)

		
		ax1 = fig.add_subplot(2, 5, cnt, projection='3d')
		ax1.set_title(all_classes[category])
		
		ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata, c=ydata)
		x, y = np.ogrid[0:gridmap.shape[0], 0:gridmap.shape[1]]
		gridmap = gridmap.astype('float32')/255
		ax1.plot_surface(x, y, np.atleast_2d(0), rstride=10, cstride=10, facecolors=gridmap)

		#ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata,label='fit', facecolors =gridmap)
		ax1.invert_yaxis()
		ax1.legend()
		cnt += 1

	plt.show()
	plt.close()
			
	


def estimateVarinace(gmap, gt_objs):

	categories = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05

	for category in range(14):
		relObjs = FindAllObjectsByCategoty(gt_objs, category)

		#avg_ = np.array()
		h, w = 400, 400
		heatmap = np.zeros((h, w), dtype="float32")
		im_cat = heatmap.flatten()

		for gt_obj in relObjs:

			df = pd.read_pickle("stats/images/heatmap2_" + str(gt_obj.uid) + ".pickle")
			# h = df['h'][0]
			# w = df['w'][0]
			im = df['data'].to_numpy()
			#print(np.unique(im))
			im_cat += im
			im[im <= 10] = 0
			

		x = range(int(-w/2), int(w/2)) 
		y = range(int(-h/2), int(h/2)) 
		xv, yv = np.meshgrid(x, y)
		xv = xv.flatten()
		yv = yv.flatten()
		xdata = np.vstack((xv, yv)).T
		ydata = np.expand_dims(im_cat, axis=1)
		nz_ids = np.nonzero(ydata)[0]
		xdata = xdata[nz_ids]
		ydata = ydata[nz_ids].squeeze()
		ydata = ydata / sum(ydata)
		if len(np.unique(ydata)) < 20:
			categories.append(category)
			mxs.append(None)
			mys.append(None)
			sxs.append(None)
			sys.append(None)
			bs.append(None)
			#corners.append(box.flatten())
			corners.append(None)
			continue


		res, mx, my, sx, sy, b, fit_y = findBestParams(xdata, ydata)

		categories.append(category)
		mxs.append(mx)
		mys.append(my)
		sxs.append(sx)
		sys.append(sy)
		bs.append(b)
		corners.append(None)


		# from mpl_toolkits import mplot3d
		# fig = plt.figure(figsize=plt.figaspect(0.5))
		# fig.suptitle(all_classes[category])
		# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
		# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
		# ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata, c=ydata, label='data')
		# ax2.scatter3D(xdata[:, 0], xdata[:, 1], fit_y, c=fit_y, label='fit');
		# ax1.legend()
		# ax2.legend()
		# plt.show()
		# plt.close()

	datadic = {'category': categories, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'b' : bs}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/variance.csv', index=False)




def createHeatMap(gmap, gt_obj, matched_preds):


	#print(objCls)
	# print(len(matched))
	h, w = gmap.map.shape
	heatmap = np.zeros(gmap.map.shape, dtype="float32")


	for obj in matched_preds:
		#print(m[0].category)
		center, dim, rot = toLabCoords(obj.center, obj.dim, obj.rot)
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz = verts[verts[:, 2].argsort()][:4]
		conf = obj.conf
		hull = ConvexHull(xyz[:, :2])
		box = np.zeros((4, 2), np.int32)
		if len(hull.vertices) == 3: 
			continue

		for v in range(4):
			vert = xyz[hull.vertices[v]]
			uv = gmap.world2map(vert)
			box[v] = uv

		tmp = np.zeros(gmap.map.shape, dtype="float32")
		cv2.fillConvexPoly(tmp, box, 1)
		heatmap += tmp

	cv2.imwrite("stats/images/heatmap_" + str(gt_obj.uid) + ".png", heatmap) 

	heatmap = heatmap.flatten()
	h = [h] * len(heatmap)
	w = [w] * len(heatmap)
	datadic = {'h': h, 'w': w, 'data': heatmap}
	df = pd.DataFrame(datadic)
	df.to_pickle("stats/images/heatmap_" + str(gt_obj.uid) + ".pickle")   


def createRelativeHeatMap(gt_obj, matched_preds):


	#print(objCls)
	# print(len(matched))
	h, w = 400, 400
	heatmap = np.zeros((h, w), dtype="float32")
	resolution = 0.05


	gt_center, _, gt_rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
	print(gt_obj.uid)

	for obj in tqdm(matched_preds):
		#print(m[0].category)
		center, dim, rot = toLabCoords(obj.center, obj.dim, obj.rot)
		# need to find a bining scheme
		center_ = [center[0], center[1], center[2], 1]
		rot, center = GetPredictionInGTFrame(origin, gt_center, gt_rot, center_, rot)
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, rot)
		verts = np.asarray(verts).reshape((8, 3))
		xyz = verts[verts[:, 2].argsort()][:4]
		hull = ConvexHull(xyz[:, :2])
		box = np.zeros((4, 2), np.int32)
		if len(hull.vertices) == 3: 
			continue

		for v in range(4):
			vert = xyz[hull.vertices[v]]
			box[v] = [int((10 + vert[0]) / resolution),int((10 + vert[1]) / resolution)]
		#print(box)


		tmp = np.zeros((h, w), dtype="float32")
		cv2.fillConvexPoly(tmp, box, 1)
		heatmap += tmp

	cv2.imwrite("stats/images/heatmap2_" + str(gt_obj.uid) + ".png", heatmap) 

	heatmap = heatmap.flatten()
	h = [h] * len(heatmap)
	w = [w] * len(heatmap)
	datadic = {'h': h, 'w': w, 'data': heatmap}
	df = pd.DataFrame(datadic)
	df.to_pickle("stats/images/heatmap2_" + str(gt_obj.uid) + ".pickle")  




def estimateVarinace4center(gmap, gt_objs):

	categories = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05

	for category in range(14):
		relObjs = FindAllObjectsByCategoty(gt_objs, category)

		#avg_ = np.array()
		h, w = 400, 400
		heatmap = np.zeros((h, w), dtype="float32")
		im_cat = heatmap.flatten()
		for gt_obj in relObjs:

			df = pd.read_pickle("stats/images/heatmap2_" + str(gt_obj.uid) + ".pickle")
			im = df['data'].to_numpy()
			im_cat += im
			#im[im <= 10] = 0
			

		x = range(int(-w/2), int(w/2)) 
		y = range(int(-h/2), int(h/2)) 
		xv, yv = np.meshgrid(x, y)
		xv = xv.flatten()
		yv = yv.flatten()
		pixels = np.vstack((xv, yv)).T
		ydata = np.zeros((400, 1), dtype=np.float32)
		xdata = np.array(range(400))
		for p in range(pixels.shape[0]):
			#print(pixels[p])
			d = np.linalg.norm(pixels[p])
			if d > 50:
				continue
			ydata[int(d)] += im_cat[p]	
		
		nz_ids = np.nonzero(ydata)[0]
		xdata = xdata[nz_ids]
		ydata = ydata[nz_ids].squeeze()
		ydata = ydata / sum(ydata)
		if len(ydata) < 1:
			categories.append(category)
			mxs.append(None)
			mys.append(None)
			sxs.append(None)
			sys.append(None)
			bs.append(None)
			#corners.append(box.flatten())
			corners.append(None)
			continue


		res, mx, my, sx, sy, b, fit_y = findBestParams4center(xdata, ydata)

		categories.append(category)
		mxs.append(mx)
		mys.append(my)
		sxs.append(sx)
		sys.append(sy)
		bs.append(b)
		corners.append(None)


		# from mpl_toolkits import mplot3d
		# fig = plt.figure(figsize=plt.figaspect(0.5))
		# fig.suptitle(all_classes[category])
		# ax1 = fig.add_subplot(1, 2, 1)
		# ax2 = fig.add_subplot(1, 2, 2)
		# ax1.scatter(xdata, ydata, label='data')
		# ax2.scatter(xdata, fit_y, label='fit');
		# ax1.legend()
		# ax2.legend()
		# plt.show()
		# plt.close()
		

	datadic = {'category': categories, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'b' : bs}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/variance.csv', index=False)






def estimateGlobalVariance4center(gmap, gt_objs):

	categories = []
	uid = []
	mxs = []
	mys = []
	sxs = []
	sys = []
	bs = []
	corners = []
	resolution = 0.05

	df = pd.read_csv('stats/variance.csv')


	for category in range(14):
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((280 * 521))

		b = df_cat['b'].to_numpy()[0]
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:

			center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			r = R.from_matrix(rot)
			roll, pitch, yaw = r.as_euler('xyz', degrees=False)
			R2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]

			box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
			verts, faces = get_cuboid_verts_faces(box3d, rot)
			verts = np.asarray(verts).reshape((8, 3))
			xyz = verts[verts[:, 2].argsort()][:4]
			hull = ConvexHull(xyz[:, :2])
			box = np.zeros((4, 2), np.int32)
		

			for v in range(4):
				vert = xyz[hull.vertices[v]]
				uv = gmap.world2map(vert)
				box[v] = uv

			categories.append(gt_obj.category)
			uid.append(gt_obj.uid)
			mxs.append(mx_)
			mys.append(my_)
			sxs.append(sx_)
			sys.append(sy_)
			bs.append(b)
			corners.append(box.flatten().tolist())	

	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)
	#df.to_pickle(output_dir + "/variance.pickle")   
	df.to_csv('stats/global_variance.csv', index=False)
	print(corners)

	import json 
	datadic = {'category': categories, 'uid': uid, 'mx' : mxs, 'my': mys, 'sx': sxs, 'sy' : sys, 'b' : bs, 'corners' : corners}
	df = pd.DataFrame(datadic)

	with open('stats/global_variance.json', 'w', encoding='utf-8') as f:
		json.dump(datadic, f, indent=4)





def createRelativeHeatMap4Center(gt_obj, matched_preds):


	#print(objCls)
	# print(len(matched))
	h, w = 400, 400
	heatmap = np.zeros((h, w), dtype="float32")
	resolution = 0.05


	gt_center, _, gt_rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
	#print(gt_obj.uid)

	for obj in tqdm(matched_preds):
		#print(m[0].category)
		center, dim, rot = toLabCoords(obj.center, obj.dim, obj.rot)
		# need to find a bining scheme
		center_ = [center[0], center[1], center[2], 1]
		rot, center = GetPredictionInGTFrame(origin, gt_center, gt_rot, center_, rot)
		
		uv = [int((10 + center[0]) / resolution),int((10 + center[1]) / resolution)]
		if uv[0] >= 0 and uv[1] >= 0 and uv[0] < w and uv[1] < h:
			heatmap[uv[1], uv[0]] += 1.0


	cv2.imwrite("stats/images/heatmap2_" + str(gt_obj.uid) + ".png", heatmap) 

	heatmap = heatmap.flatten()
	h = [h] * len(heatmap)
	w = [w] * len(heatmap)
	datadic = {'h': h, 'w': w, 'data': heatmap}
	df = pd.DataFrame(datadic)
	df.to_pickle("stats/images/heatmap2_" + str(gt_obj.uid) + ".pickle")   


def plotGlobalVariane4center(gmap, gt_objs):

	df = pd.read_csv('stats/variance.csv')
	resolution = 0.05

	def gauss(data, m, s):
		return 1.0 / np.sqrt((2. * np.pi * s**2)) * np.exp(-((data - m)**2 / (2. * s**2)))



	xdata = np.array(range(400))


	fig = plt.figure(figsize=plt.figaspect(0.5))

	cnt = 1
	for category in [0, 2, 3, 4, 6, 7, 8, 9, 11, 13]:
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		df_cat = df[df['category'] == category]
		ydata = np.zeros((400, 1))

		b = df_cat['b'].to_numpy()
		mx = df_cat['mx'].to_numpy()[0]
		my = df_cat['my'].to_numpy()[0]
		sx = df_cat['sx'].to_numpy()[0]
		sy = df_cat['sy'].to_numpy()[0]
		print(b, mx, my, sx, sy)

		for gt_obj in relObjs:

			center, dim, rot = toLabCoords(gt_obj.center, gt_obj.dim, gt_obj.rot)
			center_2d = gmap.world2map(center)
			mx_ = mx + center_2d[0]
			my_ = my + center_2d[1]
			s = np.array([[sx, 0.0], [0.0, sy]])
			s_ = R2d @ s @ R2d.T
			sx_ = s_[0][0]
			sy_ = s_[1][1]
			ydata += gauss(xdata, mx_, my_, sx_, sy_, b)
			#print(gt_obj.uid, center_2d)

		
		ax1 = fig.add_subplot(2, 5, cnt, projection='3d')
		ax1.set_title(all_classes[category])
		ax1.scatter3D(xdata[:, 0], xdata[:, 1], ydata, c=ydata, label='fit')
		ax1.invert_yaxis()
		ax1.legend()
		cnt += 1

	plt.show()
	plt.close()


def GetGlobalMapFromPickle(gmap, pickleFolder):

	mapObjects = LoadTrackedObjectsFromPickle(pickleFolder, name="/mapObjects_ICP2.pickle")
	semMas =  Build2DMapFromMapObjectsInLabFrame(gmap, mapObjects)
	for i in range(14):
		cv2.imwrite("stats/SemMaps/{}.png".format(all_classes[i]) , semMas[i])
	estimateGlobalVarianceInLabFrame(gmap, mapObjects)


def GetGlobalMapFromJSON(gmap, gtObjs):

	semMas =  Build2DMapFromMapObjects(gmap, gtObjs)
	for i in range(14):
		cv2.imwrite("stats/SemMaps/{}.png".format(all_classes[i]) , semMas[i])
	estimateGlobalVarianceInLabFrame(gmap, mapObjects)	


def main():

	gtJsonPath = "/home/nickybones/Code/Omni3DDataset/stats/Hypersim_test.json"
	predJsonPath = "/home/nickybones/Code/Omni3DDataset/stats/predictions_stats_model2.5.json"

	jsonPath = "/home/nickybones/Code/OmniNMCL/ros1_ws/src/omni3d_ros/src/Lab3D-v2.0.json"
	objs = loadObjects(jsonPath)
	gt_objs =  loadGTObjects(origin, jsonPath)
	matched_preds = {}
	for i in range(len(gt_objs)):
		matched_preds[i] = []

	gridmap = cv2.imread("/home/nickybones/Code/OmniNMCL/ros1_ws/src/omni3d_ros/configs/Map.png")
	gmap = GMAP(gridmap, 0.05, [-13.9155, -11.04537, 0.0])


	# semMas =  Build2DMapFromMapObjects(gmap, gt_objs)
	# for i in range(14):
	# 	cv2.imwrite("stats/SemMaps/{}.png".format(all_classes[i]) , semMas[i])
	#estimateVarinace4center(gmap, gt_objs)
	#exit()
	#estimateVarinace(gmap, gt_objs)
	#plotGlobalVariane(gmap, gt_objs)
	#estimateGlobalVariance(gmap, gt_objs)
	GetGlobalMapFromPickle(gmap, "/home/nickybones/Code/OmniNMCL/ros1_ws/src/omni3d_ros/")
	mapObjects = LoadTrackedObjectsFromPickle("/home/nickybones/Code/OmniNMCL/ros1_ws/src/omni3d_ros/", name="/mapObjects_ICP2.pickle")
	#plotGlobalVarianeInLabFrame(gmap, mapObjects)
	icpgrid = Build2DBordersFromMapObjectsInLabFrame(gmap, mapObjects)
	#icpgrid = Build2DBordersFromMapObjects(gmap, gt_objs)
	cv2.imwrite("map.png", icpgrid)
	exit()
	images, pr_objs = LoadObjectsFromJSON(predJsonPath)
	
	
	cnt = 0

	occ_cnt = np.zeros(14, dtype=np.float32)
	mtc_cnt = np.zeros(14, dtype=np.float32)

	for o in tqdm(pr_objs):
		image_id = o['image_id']
		category = o['category_id']
	
		gt_ = FindGTByImageID(images, image_id)
		gt = copy.deepcopy(gt_)
		im = FindImageByImageID(images, image_id)
		imOldPath = im['file_path']
		c = getCamNme(imOldPath)

		mObj = JSONObjects2MapObject(o)
		center, dim, rot = moveToGlobal(origin, gt, mObj.center, mObj.dim, mObj.rot)
		mObj.center = center
		mObj.rot = rot
		relObjs = FindAllObjectsByCategoty(gt_objs, category)
		
		best_id, best_iou = MatchGT2Pred(mObj, relObjs, gt)
		if best_id != -1:
			uid = relObjs[best_id].uid
			matched_preds[uid].append(mObj)
			mtc_cnt[category] += 1

		
		occ_cnt[category] += 1

		# cnt +=1
		# if cnt > 2000:
		# 	break

	print(mtc_cnt)
	print(occ_cnt)	
	print(mtc_cnt / occ_cnt)


	#for gt_obj in gt_objs:
		#createHeatMap(gmap, gt_obj, matched_preds[gt_obj.uid])
		#createRelativeHeatMap(gt_obj, matched_preds[gt_obj.uid])
		#createRelativeHeatMap4Center(gt_obj, matched_preds[gt_obj.uid])

	




if __name__ == "__main__":
    main()