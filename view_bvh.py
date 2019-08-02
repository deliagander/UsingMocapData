import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
# import main.py as m
import os
import numpy as np
import re
from math import sin, cos


def readBVH(mocap):

	heirarchy = np.array([])
	heirarchyEnd = np.array([])
	offsets = np.array([])
	lines = np.array([])
	isJoint = False
	endCount = 0

	for i, line in enumerate(mocap):
		lines = np.append(lines,line)

	for i, line in enumerate(lines):
		words = np.array(re.findall(r'\w+', line))
		if words.size >= 2:
			if words[0] == 'ROOT':
				bone = words[1]
				heirarchy = np.append(heirarchy,bone)
				heirarchyEnd = np.append(heirarchyEnd,bone)
				isJoint = True
			elif words[0] == 'JOINT':
				bone = words[1]
				heirarchy = np.append(heirarchy,bone)
				heirarchyEnd = np.append(heirarchyEnd,bone)
				isJoint = True
			elif words[0] == 'End':
				bone = words[0]+heirarchyEnd[-1]
				heirarchyEnd = np.append(heirarchyEnd,bone)
				endCount = endCount + 1
			elif words[0] == 'OFFSET':
				print(bone,'words', line)
				# if isJoint:
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

	offsets = offsets.reshape(heirarchyEnd.size,3)
	print(heirarchyEnd)
	return translations, rotations, heirarchy, offsets, heirarchyEnd

# Source for rotation functions: https://github.com/matt-graham/bvh-tools/blob/master/bvh/numpy_renderer.py

def rot_x(ang):
    return np.array([
        [1., 0., 0.], [0., cos(ang), -sin(ang)], [0., sin(ang), cos(ang)]
    ])


def rot_y(ang):
    return np.array([
        [cos(ang), 0., sin(ang)], [0., 1., 0.], [-sin(ang), 0., cos(ang)]
    ])


def rot_z(ang):
    return np.array([
        [cos(ang), -sin(ang), 0.], [sin(ang), cos(ang), 0.], [0., 0., 1.]
    ])


rotation_map = {
    'Xrotation': rot_x,
    'Yrotation': rot_y,
    'Zrotation': rot_z
}

def moveVertices(translation, rotation, offsets, heirarchy, parents, heirarchyEnd, children, indices):

	compositeTransformations = {}

	rotation = rotation * np.pi / 180
	vertices = np.zeros((heirarchyEnd.size,3))

	for i, joint in enumerate(heirarchyEnd):
		rot = np.eye(3)

		parent = parents[joint]
		child = children[joint]

		if child == -1:
			parent_str = heirarchyEnd[parent]
			offset = offsets[i,:]
			parent_trans = compositeTransformations[parent_str]
			node_trans = parent_trans
			point_1 = node_trans.dot(np.r_[offset, 1.])
			vertices[i,:] = point_1[:3]

		else:
			index = indices[joint]

			for c, ch in enumerate(['Yrotation', 'Xrotation', 'Zrotation']):
				# if ch in rotation_map:
				rot = rot.dot(rotation_map[ch](rotation[index,c]))

			local_trans = np.zeros((4, 4))
			local_trans[:3, :3] = rot

			if parent == -1:
				# local_trans[:, 3] = np.r_[translation, 1.]
				local_trans[:, 3] = [0.,0.,0., 1.]
				node_trans = local_trans
				compositeTransformations[joint] = node_trans
				point_1 = node_trans.dot(np.array([0., 0., 0., 1.]))
				print(node_trans)

				print(point_1)
				print(vertices.shape)
				vertices[i,:] = point_1[:3]		

			else:
				parent_str = heirarchyEnd[parent]
				index = indices[joint]
				local_trans[:, 3] = np.r_[offsets[i,:], 1.]
				parent_trans = compositeTransformations[parent_str]
				node_trans = parent_trans.dot(local_trans)
				compositeTransformations[joint] = node_trans
				point_1 = node_trans.dot(np.array([0., 0., 0., 1.]))
				vertices[i,:] = point_1[:3]
	return vertices

def draw(vertices, edges):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def main():
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	mocap  = open(os.path.join(__location__,'VFX_Students_Shoot_001.bvh'), 'r') 

	translations, rotations, heirarchy, offsets, heirarchyEnd = readBVH(mocap)
	# print('translations', translations.shape)
	print('offsets', offsets)
	# print(heirarchy)
	# print('rotations', rotations[2,:,:])

	# bones = main.readObj(model, heirarchy)

	parents = {'Hips': -1,
			'Chest': 0,
			'Chest2': 1,
			'Chest3': 2,
			'Chest4': 3,
			'Neck': 4,
			'Head': 5,
			'EndHead': 6,
			'RightCollar': 4,
			'RightShoulder':8,
			'RightElbow':9,
			'RightWrist': 10,
			'Prop1': 11,
			'EndProp1': 12,
			'EndEndProp1': 11,
			'LeftCollar': 4,
			'LeftShoulder': 15,
			'LeftElbow': 16,
			'LeftWrist': 17,
			'EndLeftWrist': 18,
			'RightHip': 0,
			'RightKnee': 20,
			'RightAnkle': 21,
			'RightToe': 22,
			'EndRightToe': 23,
			'LeftHip': 0,
			'LeftKnee': 25,
			'LeftAnkle': 26,
			'LeftToe': 27,
			'EndLeftToe': 28}

	child = {'Hips': [1,20,25],
				'Chest': [2],
				'Chest2': [3],
				'Chest3': [4],
				'Chest4': [5,8,15],
				'Neck': [6],
				'Head': [7],
				'EndHead': -1,
				'RightCollar': [9],
				'RightShoulder':[10],
				'RightElbow':[11],
				'RightWrist': [12],
				'Prop1': [13],
				'EndProp1': -1,
				'EndEndProp1': -1,
				'LeftCollar': [16],
				'LeftShoulder': [17],
				'LeftElbow': [18],
				'LeftWrist': [19],
				'EndLeftWrist': -1,
				'RightHip': [21],
				'RightKnee': [22],
				'RightAnkle': [23],
				'RightToe': [24],
				'EndRightToe': -1,
				'LeftHip': [26],
				'LeftKnee': [27],
				'LeftAnkle': [28],
				'LeftToe': [29],
				'EndLeftToe':-1}

	indices = {'Hips': 0,
			'Chest': 1,
			'Chest2': 2,
			'Chest3': 3,
			'Chest4': 4,
			'Neck': 5,
			'Head': 6,
			'RightCollar': 7,
			'RightShoulder':8,
			'RightElbow':9,
			'RightWrist': 10,
			'Prop1': 11,
			'LeftCollar': 12,
			'LeftShoulder': 13,
			'LeftElbow': 14,
			'LeftWrist': 15,
			'RightHip': 16,
			'RightKnee': 17,
			'RightAnkle': 18,
			'RightToe': 19,
			'LeftHip': 20,
			'LeftKnee': 21,
			'LeftAnkle': 22,
			'LeftToe': 23}

	cumulativeOffsets = {}
	vertices = np.array([])
	edges = np.array([])

	for i, joint in enumerate(heirarchyEnd):
		print('joint', joint)
		parent = parents[joint]
		print('parent', heirarchyEnd[parent])
		if parent == -1:
			cumulativeOffsets[joint] = offsets[i,:]
			vertices = np.append(vertices,cumulativeOffsets[joint])
		else:
			parent_str = heirarchyEnd[parent]
			cumulativeOffsets[joint] = offsets[i,:] + cumulativeOffsets[parent_str]
			vertices = np.append(vertices,cumulativeOffsets[joint])
			edges = np.append(edges,np.array([parent,i]))

	vertices = vertices.reshape(heirarchyEnd.size,3)/100
	print('edge size',edges.size)
	edges = edges.reshape(int(edges.size/2),2).astype(int)

	print('vert',vertices)
	print('edge',edges)

	frame = 500

	translation = translations[frame, :] 
	rotation = rotations[frame, :, :] 

	vertices = moveVertices(translation, rotation, offsets, heirarchy, parents, heirarchyEnd, child, indices)

	vertices = vertices/100
	# print('vert',vertices)

	pygame.init()
	display = (800,600)
	pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

	gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

	glTranslatef(0.0,0.0, -5)

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

        # glRotatef(1, 3, 1, 1)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		draw(vertices, edges)
  #       glBegin(GL_LINES);
  #   		glVertex2f(10, 10);
  #   		glVertex2f(20, 20);
		# glEnd();

		pygame.display.flip()
		pygame.time.wait(10)


main()