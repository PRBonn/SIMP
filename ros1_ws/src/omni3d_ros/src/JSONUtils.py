
import os
import json

def FindGTByImageID(images, image_id):

	for im in images:
		if im['id'] == image_id:
			return im['gt']


	return None


def FindLastImageFrameNum(images):

	im = images[-1]
	imOldPath = im['file_path']
	camPath = os.path.split(imOldPath)[1].replace('.png', '')
	
	return int(camPath)



	return None

def FindImageByImageID(images, image_id):

	for im in images:
		if im['id'] == image_id:
			return im


	return None

def FindAllImagseBySeqID(images, seq_id):

	seq_images = []
	for im in images:
		if im['seq_id'] == seq_id:
			seq_images.append(im)


	return seq_images

def FindAllImagseByFrameID(images, frame_id):

	frame_images = []
	for im in images:
		if "/{:05d}.png".format(frame_id) in im['file_path']:
			frame_images.append(im)


	return frame_images

def GetAllSequences(images):

	seq_ids = set()

	for im in images:
		seq_ids.add(im['seq_id'])

	return list(seq_ids)


def FindAllObjectsByImageID(objects, image_id):

	matches = []

	for o in objects:
		if o['image_id'] == image_id:
			matches.append(o)


	return matches

def LoadObjectsFromJSON(jsonPath):

	with open(jsonPath) as f:
		data = json.load(f)


	images = data['images']
	objects = data['annotations']

	return images, objects



def getCamNmeFromImage(im):

	seq = im['seq_id']
	imOldPath = im['file_path']
	c = getCamNme(imOldPath)

	return c


def getCamNme(imOldPath):

	camPath = os.path.split(imOldPath)[0]
	camPath = os.path.split(camPath)[1]
	
	if '00' in camPath:
		return 0

	elif '01' in camPath:
		return 1

	elif '02' in camPath:
		return 2

	elif '03' in camPath:
		return 3

	else:
		return -1

def retrieveOrigImagePath(im):

	seq = im['seq_id']
	imOldPath = im['file_path']
	imName = os.path.split(imOldPath)[1]
	c = getCamNme(imOldPath)
	imPath = "R{}/camera{}/color/{}".format(seq,c,imName)
	
	return imPath


def getCategories(predJsonPath):

	with open(predJsonPath) as f:
		data = json.load(f)

	cats = data['categories']

	classes = []

	for cat in cats:
		classes.append(cat['name'])

	return classes