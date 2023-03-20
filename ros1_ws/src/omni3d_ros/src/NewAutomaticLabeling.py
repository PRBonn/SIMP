import json 
from GMAP import GMAP
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
import copy
import pandas as pd
import open3d as o3d

from Lab3DObject import Lab3DObject
from Omni3DObject import Omni3DObject
from Omni3DImage import Omni3DImage
from Omni3DCategory import Omni3DCategory
from Omni3DDataset import Omni3DDataset
from Omni3DInfo import Omni3DInfo
from DatasetUtils import *
from VisUtils import *
from ObjectUtils import *
from JSONUtils import *



all_classes = ['sink', 'door', 'oven', 'board', 'table', 'box', 'potted plant', 'drawers', 'sofa', 'cabinet', 'chair', 'fire extinguisher', 'person', 'desk']
selected_classes = ['sink', 'oven', 'board', 'table', 'potted plant', 'drawers', 'sofa', 'cabinet', 'fire extinguisher', 'desk']

omni_classes = ['chair', 'table', 'cabinet', 'car', 'lamp', 'books', 'sofa', 'pedestrian', 'picture', 'window', 'pillow', 'truck', 'door', 'blinds', 'sink', 'shelves', 'television', 'shoes', 'cup', 'bottle', 'bookcase', 'laptop', 'desk', 'cereal box', 'floor mat', 'traffic cone', 'mirror', 'barrier', 'counter', 'camera', 'bicycle', 'toilet', 'bus', 'bed', 'refrigerator', 'trailer', 'box', 'oven', 'clothes', 'van', 'towel', 'motorcycle', 'night stand', 'stove', 'machine', 'stationery', 'bathtub', 'cyclist', 'curtain', 'bin']

cam_poses = np.array([[0.1, 0, 0], [0, -0.1, -0.5 * np.pi], [-0.1, 0, np.pi], [0, 0.1, 0.5 * np.pi]])

camere_z = 0.63
min_z = -1.55708286

img_w = 640
img_h = 480



origin = o3d.geometry.TriangleMesh.create_coordinate_frame()


def plotObjs(objs, gt=None):

	cubes = []
	colors = []
	clr = cm.rainbow(np.linspace(0, 1, 14))

	for o in objs:
		center = o.center
		dim = o.dim
		box3d = [center[0], center[1], center[2], dim[0], dim[1], dim[2]]
		verts, faces = get_cuboid_verts_faces(box3d, o.rot)
		xyz = np.asarray(verts).reshape((8, 3))
		cubes.append(xyz)
		colors.append(clr[o.category][:3])

	if gt:
		box3d = [gt[0], gt[1], gt[2], 2.0, 0.5, 0.1]
		verts, faces = get_cuboid_verts_faces(box3d, origin.get_rotation_matrix_from_xyz((0, -gt[3], 0)))
		xyz = np.asarray(verts).reshape((8, 3))
		cubes.append(xyz)
		colors.append([0, 0, 0])

	debug3DViz(origin, cubes, colors)


def AutomaticLabeling(folderPath, sequences, gmap, objs, semMaps, k, cam_trans, split="All", debug=False):


	#create categories
	categories = []
	for i, c in enumerate(all_classes):
		ct = Omni3DCategory(i, c)
		categories.append(ct)

	dataset_id = 11
	images = []
	annotations = []
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	uniq_img_id = 0
	step = 10
	if split == "Validation" or split == "Test":
		step = 20

	for seq in sequences:

		subfolderPath = folderPath + "R{}/".format(seq)
		#df = pd.read_pickle(subfolderPath + "icra2023_R{}.pickle".format(seq))
		df = pd.read_pickle(subfolderPath + "omniData_R{}.pickle".format(seq))
		print("seq R{}".format(seq))

		for cam in range(4):
			print("cam {}".format(cam))
			sem = "sem{}".format(cam)
			df_sem = df[df['type'] == sem]
			if df_sem.empty:
				continue
			df_gt = df[df['type'] == 'gt']
			df_cam = pd.read_pickle(subfolderPath + "camera.pickle")
			clr = cm.rainbow(np.linspace(0, 1, len(all_classes)))

			img_path  = subfolderPath + "/camera{}/color/".format(cam) + df_cam.iloc[0]['data']
			img = cv2.imread(img_path)
			img_h, img_w, img_c = img.shape


			for index, row in df_sem.iterrows(): 

				if index % step:
					continue

				t_img = row['t']
				rel_img_path = "/camera{}/color/".format(cam) + df_cam.iloc[(df_cam['t']-t_img).abs().argsort()[:1]]['data'].to_numpy()[0]
				img_path  = subfolderPath + "/camera{}/color/".format(cam) + df_cam.iloc[(df_cam['t']-t_img).abs().argsort()[:1]]['data'].to_numpy()[0]
				img = cv2.imread(img_path)
				# if seq == 15:
				# 	img = cv2.rotate(img, cv2.ROTATE_180)

				file_path = "hypersim/R{}/images/scene_cam_0{}_final_preview/".format(seq, cam) + df_cam.iloc[(df_cam['t']-t_img).abs().argsort()[:1]]['data'].to_numpy()[0]

				# get gt pose
				tmp = df_gt.iloc[(df_gt['t']-t_img).abs().argsort()[:1]]['data'].to_numpy()[0]
				gt_2d = copy.deepcopy(tmp)
				#gt_2d = df_gt.iloc[(df_gt['t']-t_img).abs().argsort()[:1]]['data'].to_numpy()[0]
				#gt_trans  = trans2d(0.1, 0, gt_2d[2])
				gt_trans = v2t(gt_2d)
				cam_pos = np.array([cam_poses[cam]]).T
				cam_pos[2] = 1.0
				gt_trans = gt_trans @ cam_pos
				gt_2d[0] = gt_trans[0].item()
				gt_2d[1] = gt_trans[1].item()
				gt_2d[2] += cam_poses[cam][2]
				
				# assumigng the YouBot height, need to verify
				gt = [gt_2d[0], gt_2d[1], min_z + camere_z, gt_2d[2]]
				gt = gt2Omni3DCoords(origin, gt)
			
			
				data = row['data']
				numDetect = int(len(data) / 6)

				cubes = []
				colors = []
				valid = False
				temp_img = img.copy()

				tempAnnotations = []
				for d in range(numDetect):

					category = int(data[d * 6])
					color = 255 * clr[category][:3]

					if all_classes[category] in selected_classes:

						iou_fin = 0
						gid_fin = -1

						omniObject_fin = None

						u1, v1 = int(data[d * 6 + 1]), int(data[d * 6 + 2])
						u2, v2 = int(data[d * 6 + 3]), int(data[d * 6 + 4])
						bbox2D_tight = [u1, v1, u2, v2]
						conf = data[d * 6 + 5]
						if conf < 0.7:
							continue

						#print(all_classes[category])

						cv2.rectangle(temp_img, (u1, v1), (u2, v2), color, 2)
						cv2.putText(temp_img, "{} {:.2f}".format(all_classes[category], conf), (u1, v1 +10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

						tracedObjs = getTraceSameClassObjects(objs, category, gmap, gt, semMaps)
						if not tracedObjs:
							continue
						print("here")
						im_rendered, fragment, meshes, color_ids = renderMeshes(origin, k, img_w, img_h, tracedObjs, gt, device, viz=False)
						im_rendered = im_rendered.squeeze()[:, :, :3].cpu().numpy()
						cv2.imshow("", im_rendered)
						cv2.waitKey()


						for oid, obj in enumerate(tracedObjs):

							center = obj.center
							dimensions = [obj.dim[0], obj.dim[1], obj.dim[2] ] 
							rot = obj.rot

							R_cam, center_cam = getComposedTransformOmni(origin, gt, [center[0], center[1], center[2], 1], rot)
							center_cam = center_cam.flatten()

							box3d = [center_cam[0], center_cam[1], center_cam[2], dimensions[0], dimensions[1], dimensions[2]]
							verts, faces = get_cuboid_verts_faces(box3d, R_cam)
							bbox3D_cam = np.asarray(verts).reshape((8, 3))

							truncation = estimate_truncation(k, box3d, rot, img_w, img_h)

							if truncation < 0.66:
								_, fragment, _, _ = renderMeshes(origin, k, img_w, img_h, [obj], gt, device, viz=False)
								viz = visibilityAnalysis(im_rendered, obj, fragment, color_ids[oid], img_h, img_w)
								#print(viz)
								if viz:							
									
									xyxy, behind_camera, fully_behind = convert_3d_box_to_2d(k, box3d, R_cam, clipw=img_w, cliph=img_h, XYWH=False, min_z=0.00)
									if fully_behind:
										continue

									bbox2D_proj = xyxy.cpu().detach().numpy().astype(np.int32)
									bbox2D_trunc =  getTrunc2Dbbox(bbox2D_proj, img_h, img_w)
									iou =  IoU(bbox2D_tight[:2], bbox2D_tight[2:], bbox2D_trunc[:2], bbox2D_trunc[2:])
									if iou > 0.25:					
										valid = True
										if iou > iou_fin:
											iou_fin = iou
											gid_fin = obj.uid

											visibility = estimate_visibility(k, box3d, R_cam, img_w, img_h, device=device)[0]
											behind_camera = behind_camera.cpu().detach().numpy()

											omniObject_fin =  Omni3DObject(dataset_id, uniq_img_id, seq, category, all_classes[category], 1, \
												bbox2D_tight, bbox2D_proj.tolist(),  #change this to 2D corners projected from bbox3D
												bbox2D_trunc.tolist(),  #change this to 2D corners projected from bbox3D then truncated
												bbox3D_cam.tolist(), center_cam.tolist(), dimensions, R_cam.tolist(), int(behind_camera), visibility, truncation)

						if omniObject_fin: 

							#print(omniObject_fin.toJSON())

							tempAnnotations.append(omniObject_fin)
							xyz = omniObject_fin.bbox3D_cam
							xyz = np.asarray(xyz).reshape((8, 3))
							
							cubes.append(xyz)
							colors.append(clr[category][:3])
							u1, v1 , u2, v2 = omniObject_fin.bbox2D_proj
							cv2.rectangle(temp_img, (u1, v1), (u2, v2), color, 2)
							#print(gid_fin, iou_fin, omniObject_fin.truncation, omniObject_fin.visibility)



				if valid:
					#create image object
					debug3DViz(origin, cubes, colors)
					print("image {}".format(index))
					cv2.imshow("Labels", temp_img)
					key = cv2.waitKey()
					# if debug:
					# 	debug3DViz(origin, cubes, colors)

				imgObj = Omni3DImage(uniq_img_id, dataset_id, seq, img_w, img_h, file_path, k.tolist(), gt, t_img)
				images.append(imgObj)
				annotations.extend(tempAnnotations)
				uniq_img_id += 1
				
				#break


	print("Dumping to json files")
	if split == "Train" or split == "All":
		info = Omni3DInfo(str(dataset_id), 0, "Hypersim", "Train" , "0.1", "")
		dataset = Omni3DDataset(info, images, categories, annotations)
		with open("Hypersim_train.json", "w") as f:
			f.write("{}".format(dataset.toJSON()))

	if split == "Validation" or split == "All":
		info = Omni3DInfo(str(dataset_id), 0, "Hypersim", "Validation" , "0.1", "")
		dataset = Omni3DDataset(info, images, categories, annotations)
		with open("Hypersim_val.json", "w") as f:
			f.write("{}".format(dataset.toJSON()))

	if split == "Test" or split == "All":
		info = Omni3DInfo(str(dataset_id), 0, "Hypersim", "Test" , "0.1", "")
		dataset = Omni3DDataset(info, images, categories, annotations)
		with open("Hypersim_mapping2.json", "w") as f:
			f.write("{}".format(dataset.toJSON()))

		


def testLoadGTObjects(origin, jsonPath):

	objs = loadGTObjects(origin, jsonPath)
	plotObjs(objs)

def main():

	
	with open("realsense640x480/cam0.config", 'r') as f:
		data = json.load(f)

	k = np.reshape(np.array([data['k']]), (3,3))
	cam_trans = np.linalg.inv(np.reshape(np.array([data['t']]), (3,3)))
	
	jsonPath = "Lab3D-v1.0.json"
	# testLoadGTObjects(origin, jsonPath)
	# exit()

	objs = loadGTObjects(origin, jsonPath)

	gridmap = cv2.imread("Map.png")
	gmap = GMAP(gridmap, 0.05, [-13.9155, -11.04537, 0.0])
	objMap, semMaps = draw2DMap2(gmap, objs, all_classes)
	cv2.imwrite("objMap.png", objMap)

	#folderPath = "/home/nickybones/data/MCL/omni3d/Map2/"
	folderPath = "/media/nickybones/My Passport/post_processed/omni3d/Map2/"
	#sequences = [23, 24, 25, 26, 27, 28, 17, 18, 19, 2,3, 6, 9, 11, 14]
	#sequences = [17]
	#sequences = [13, 5, 8, 10]

	#AutomaticLabeling(folderPath, sequences, gmap, objs, semMaps, k, cam_trans, "Train", False)
	#AutomaticLabeling(folderPath, sequences, gmap, objs, semMaps, k, cam_trans, "Validation", False)
	#folderPath = "/home/nickybones/data/MCL/omni3d/"
	AutomaticLabeling(folderPath, [30, 31], gmap, objs, semMaps, k, cam_trans, "Test", False)

	
	
	



if __name__ == "__main__":
    main()
