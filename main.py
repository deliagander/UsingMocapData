import numpy as np
import re
import os
import copy
from math import sin, cos


def readBVH(mocap):

	heirarchy = np.array([])
	offsets = np.array([])
	lines = np.array([])
	isJoint = False
	for i, line in enumerate(mocap):
		lines = np.append(lines,line)

	for i, line in enumerate(lines):
		words = np.array(re.findall(r'\w+', line))
		if words.size >= 2:
			if words[0] == 'ROOT':
				bone = words[1]
				heirarchy = np.append(heirarchy,bone)
				isJoint = True
			elif words[0] == 'JOINT':
				bone = words[1]
				heirarchy = np.append(heirarchy,bone)
				isJoint = True
			elif words[0] == 'OFFSET':
				print(bone,'words', line)
				if isJoint:
					offset = np.array(re.findall(r"[-+]?\d*\.*\d+", line))
					offset = offset.astype(float)
					offsets = np.append(offsets,offset)
				isJoint = False
			elif words[0] == 'Frames':
				frames = int(words[1])
				translations = np.zeros([frames,3])
				rotations = np.zeros([frames,heirarchy.size,3])
				data = lines[(i+2):]
				break

	for i, line in enumerate(data):
		numbers = np.array(re.findall(r"[-+]?\d*\.*\d+", line))
		numbers = numbers.astype(float)
		translations[i,:] = numbers[:3]
		rot = numbers[3:].reshape(heirarchy.size,3)
		rotations[i,:,:] = rot

	offsets = offsets.reshape(heirarchy.size,3)

	return translations, rotations, heirarchy, offsets

def readObj(model, heirarchy):
	bones = {}

	for joint in heirarchy:
		bones[joint] = {}

	startIndex = 0
	vertexIndex = 0
	jointIndex = 0

	vertices = np.array([])
	vertexNormals = np.array([])
	vertexTextures = np.array([])
	faces = np.array([])
	joint = np.nan

	for cnt, lines in enumerate(model):
		if (lines[0] == 'g') and ('default' in lines):
			if jointIndex > 0:
				vertices = vertices.reshape(int(vertices.size/3),3)
				bones[joint] = {'vertices': vertices,
								'normals': vertexNormals,
								'textures': vertexTextures,				
								'faces': faces}

				print('joint', joint)
				print('vertices', bones[joint]['vertices'].shape)
				print('normals', vertexNormals.shape)
				print('textures', vertexTextures.shape)
				print('faces', faces.shape)

			vertices = np.array([])
			vertexNormals = np.array([])
			vertexTextures = np.array([])
			faces = np.array([])
			joint = np.nan

			jointIndex = jointIndex+1
			# print('joint index', jointIndex)

		elif (lines[0] == 'v'):
			if (lines[1] == 'n'):
				vertexNormals = np.append(vertexNormals,lines)

			elif (lines[1] == 't'):
				vertexTextures = np.append(vertexTextures,lines)

			else:
				vertex = re.findall(r"[-+]?\d*\.*\d+", lines)
				vertex = np.array(vertex)
				vertex = vertex.astype(np.float)*10
				vertices = np.append(vertices,vertex)

		elif (lines[0] == 'f'):
			faces = np.append(faces,lines)

		elif (lines[0] == 'g') and ('default' not in lines):
			words = np.array(re.findall(r'\w+', lines))
			joint = words[1]

	vertices = vertices.reshape(int(vertices.size/3),3)
	bones[joint] = {'vertices': vertices,
					'normals': vertexNormals,
					'textures': vertexTextures,				
					'faces': faces}

	print('joint', joint)
	print('vertices', bones[joint]['vertices'].shape)
	print('normals', vertexNormals.shape)
	print('textures', vertexTextures.shape)
	print('faces', faces.shape)
	return bones


def moveVertices(bones, translation, rotation, offsets, heirarchy, parents):
	new_bones = copy.deepcopy(bones)

	compositeTransformations = {}
	reverseTranslations = {}

	for i, joint in enumerate(heirarchy):
		parent = parents[joint]
		if 'ROOT' in parent:
			reverseTranslations[joint] = [0,0,0]
		else:
			reverseTranslations[joint] = reverseTranslations[parent] - offsets[i,:]
		
		vertices = bones[joint]['vertices']
		new_vertices = np.zeros(vertices.shape)

		for v, vertex in enumerate(vertices):
			vert = vertex + reverseTranslations[joint]
			new_vertices[v, :] = vert

		new_bones[joint]['vertices'] = new_vertices


	rotation = rotation * np.pi / 180


	for i, joint in enumerate(heirarchy):
		rot_y = np.array([[cos(rotation[i,0]), 0., sin(rotation[i,0])], [0., 1., 0.], [-sin(rotation[i,0]), 0., cos(rotation[i,0])]])
		rot_x = np.array([[1., 0., 0.], [0., cos(rotation[i,1]), -sin(rotation[i,1])], [0., sin(rotation[i,1]), cos(rotation[i,1])]])
		rot_z = np.array([[cos(rotation[i,2]), -sin(rotation[i,2]), 0.], [sin(rotation[i,2]), cos(rotation[i,2]), 0.], [0., 0., 1.]])

		rot = rot_y@rot_x@rot_z

		local_trans = np.zeros((4, 4))
		local_trans[:3, :3] = rot

		parent = parents[joint]
		if 'ROOT' in parent:
			local_trans[:, 3] = np.r_[translation, 1.]
			# local_trans[:, 3] = np.r_[0.,0.,0., 1.]
			node_trans = local_trans
			compositeTransformations[joint] = node_trans
		else:
			local_trans[:, 3] = np.r_[offsets[i,:], 1.]
			# local_trans[:, 3] = np.r_[0.,0.,0., 1.]
			parent_trans = compositeTransformations[parent]
			node_trans = parent_trans.dot(local_trans)
			compositeTransformations[joint] = node_trans

		vertices = new_bones[joint]['vertices']
		new_vertices = np.zeros(vertices.shape)
			
		for v, vertex in enumerate(vertices):
			vert = node_trans@np.r_[vertex, 1.].T
			# print('new vertex', vert)
			new_vertices[v, :] = vert[:3]

		new_bones[joint]['vertices'] = new_vertices

	return new_bones

		

def saveObj(bones, frame, heirarchy):
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	file = open(os.path.join(__location__,'Frames/frame'+str(frame)+'.obj'), 'w')

	file.write("# This file uses centimeters as units for non-parametric coordinates.\n \n")
	file.write("mtllib low_poly_model.mtl\n")

	for joint in heirarchy:
		# file.write("#vertices: " + str(no_vertices) + "\n")
		# file.write("#faces: " + str(no_faces) + "\n")
		vertices = bones[joint]['vertices']
		vertexNormals = bones[joint]['normals']
		vertexTextures = bones[joint]['textures']
		faces = bones[joint]['faces']

		file.write("g default\n")

		for v in vertices:
			file.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")

		for vt in vertexTextures:
			file.write(vt)

		for vn in vertexNormals:
			file.write(vn)

		file.write("s off\n")
		file.write("g" + " " + joint + "\n")

		if 'Prop1' in joint:
			file.write("usemtl espada_low:defaultMat\n")
		else:
			file.write("usemtl human_skeleton:default1\n")

		for f in faces:
			file.write(f)

	file.close()
	print('Frame', frame, 'saved!')



if __name__ == "__main__":
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	model  = open(os.path.join(__location__,'model/low_poly_t_pose.obj'), 'r') 
	mocap  = open(os.path.join(__location__,'VFX_Students_Shoot_001.bvh'), 'r') 

	translations, rotations, heirarchy, offsets = readBVH(mocap)
	# print('offsets', offsets)

	bones = readObj(model, heirarchy)

	parents = {'Hips': 'ROOT',
			'Chest': 'Hips',
			'Chest2': 'Chest',
			'Chest3': 'Chest2',
			'Chest4': 'Chest3',
			'Neck':'Chest4',
			'Head': 'Neck',
			'RightCollar': 'Chest4',
			'RightShoulder':'RightCollar',
			'RightElbow':'RightShoulder',
			'RightWrist': 'RightElbow',
			'Prop1': 'RightWrist',
			'LeftCollar': 'Chest4',
			'LeftShoulder': 'LeftCollar',
			'LeftElbow': 'LeftShoulder',
			'LeftWrist': 'LeftElbow',
			'RightHip': 'Hips',
			'RightKnee': 'RightHip',
			'RightAnkle': 'RightKnee',
			'RightToe': 'RightAnkle',
			'LeftHip': 'Hips',
			'LeftKnee': 'LeftHip',
			'LeftAnkle': 'LeftKnee',
			'LeftToe': 'LeftAnkle'}

	# Test frames

	translation = translations[30, :]
	rotation = rotations[30, :, :]
	new_bones = moveVertices(bones, translation, rotation, offsets, heirarchy, parents)
	saveObj(new_bones, 30, heirarchy)


	translation = translations[2571, :]
	rotation = rotations[2571, :, :]
	new_bones = moveVertices(bones, translation, rotation, offsets, heirarchy, parents)
	saveObj(new_bones, 2571, heirarchy)

	# frames = translations.shape[0]
	# for i in range(1,frames):
	# 	translation = translations[i, :] 
	# 	rotation = rotations[i, :, :] 
	# 	new_bones = moveVertices(bones, translation, rotation, offsets, heirarchy, parents)
	# 	saveObj(new_bones, i, heirarchy)




