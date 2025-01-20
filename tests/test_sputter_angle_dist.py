# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.operations.sputter_angle_dist as sp_angle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as Rotate

plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def rotation_matrix(normal_theta, normal_phi):
    cos_theta = np.cos(normal_theta)
    sin_theta = np.sin(normal_theta)
    cos_phi = np.cos(normal_phi)
    sin_phi = np.sin(normal_phi)
    RZmatrix = np.array([[cos_theta, -sin_theta,  0],
                         [sin_theta,  cos_theta,  0],
                         [        0,          0,  1]])
    RXmatrix = np.array([[        1,          0,        0],
                         [        0,    cos_phi, -sin_phi],
                         [        0,    sin_phi,  cos_phi]])
    return np.dot(RZmatrix, RXmatrix)
    # return np.dot(RXmatrix, RZmatrix)

def rotationRZ(normal_theta):
    cos_theta = np.cos(normal_theta)
    sin_theta = np.sin(normal_theta)
    RZmatrix = np.array([[cos_theta, -sin_theta,  0],
                         [sin_theta,  cos_theta,  0],
                         [        0,          0,  1]])
    return RZmatrix

def rotationRX(normal_phi):
    cos_phi = np.cos(normal_phi)
    sin_phi = np.sin(normal_phi)
    RXmatrix = np.array([[        1,          0,        0],
                         [        0,    cos_phi, -sin_phi],
                         [        0,    sin_phi,  cos_phi]])
    return RXmatrix

def rotationRY(normal_phi):
    cos_phi = np.cos(normal_phi)
    sin_phi = np.sin(normal_phi)
    RXmatrix = np.array([[  cos_phi,  0, sin_phi],
                         [        0,    1, 0],
                         [ -sin_phi,    0,  cos_phi]])
    return RXmatrix

theta, phi = np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi/2, np.pi/2, 100)
THETA, PHI = np.meshgrid(theta, phi)
# R = np.cos(PHI**2)
inject_angle = np.pi/12

R = sp_angle.emission_angle( THETA, PHI, theta1 = inject_angle, Eth=26.8, E = 50)

X = np.zeros_like(R)
Y = np.zeros_like(R)
Z = np.zeros_like(R)


vel = np.array([-1, 1, -1])
vel = np.divide(vel, np.linalg.norm(vel, axis=-1))
phi0 = np.arctan2(vel[1], vel[0]) - np.pi/2

# normal = np.array([1,0, 1])
normal = np.array([1, 0, 3**0.5])
normal = np.divide(normal, np.linalg.norm(normal, axis=-1))

# normal_t = np.arccos(normal[2])
normal_t = np.arccos(np.dot(normal, np.array([0, 0, 1])))

# normal_t = np.pi/6
normal_t = 0
# normal_p = np.arctan2(normal[1], normal[0])
r = Rotate.from_rotvec([0, normal_t, 0])
# normal_p = np.arccos(normal[2])
# normal_t = np.arctan2(normal[1], normal[0])
print('theta0',normal_t/np.pi *180)
print('phi0', phi0/np.pi *180)
# print(normal_p)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        # xyz = np.dot(rotationRY(normal_t), np.array([R[i,j] * np.sin(THETA[i,j]) * np.cos(PHI[i,j]), 
        #                 R[i,j] * np.sin(THETA[i,j]) * np.sin(PHI[i,j]), 
        #                 R[i,j] * np.cos(THETA[i,j])]))
        # xyz = rotationRY(normal_t)@np.array([R[i,j] * np.sin(THETA[i,j]) * np.cos(PHI[i,j]), 
        #                 R[i,j] * np.sin(THETA[i,j]) * np.sin(PHI[i,j]), 
        #                 R[i,j] * np.cos(THETA[i,j])])
        xyz = r.apply(np.array([R[i,j] * np.sin(THETA[i,j]) * np.cos(PHI[i,j]), 
                        R[i,j] * np.sin(THETA[i,j]) * np.sin(PHI[i,j]), 
                        R[i,j] * np.cos(THETA[i,j])]))
        # xyz = np.dot(xyz, rotationRZ(phi0))
        # print(xyz.shape)
        X[i,j] = xyz[0]
        Y[i,j] = xyz[1]
        Z[i,j] = xyz[2]
        # X[i,j] = R[i,j] * np.sin(THETA[i,j]) * np.cos(PHI[i,j])*rotation_matrix(np.pi/4)
        # Y[i,j] = R[i,j] * np.sin(THETA[i,j]) * np.sin(PHI[i,j])*rotation_matrix(np.pi/4)
        # # Z = np.abs(R * np.cos(THETA))
        # Z[i,j] = R[i,j] * np.cos(THETA[i,j])*rotation_matrix(np.pi/4)

# X = R * np.sin(THETA) * np.cos(PHI)*rotation_matrix(np.pi/4)
# Y = R * np.sin(THETA) * np.sin(PHI)*rotation_matrix(np.pi/4)
# # Z = np.abs(R * np.cos(THETA))
# Z = R * np.cos(THETA)*rotation_matrix(np.pi/4)
# renormal = Z < 0
# Z[renormal] = 0
    
# rotation_matrix(np.pi/4, np.pi/4)


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    linewidth=0, antialiased=False, alpha=0.2)

# inject_angle_array = np.array([-1, 0, -np.tan(np.pi/2-inject_angle)])
# inject_angle_array = np.divide(inject_angle_array, np.linalg.norm(inject_angle_array, axis=-1))
# inject_angle_rotate = np.dot(inject_angle_array, rotation_matrix(normal_t, normal_p))
# inject_normal_rotate = np.dot(np.array([0, 0, -1]), rotation_matrix(normal_t, normal_p))

print('angle', inject_angle)
# print('angle between:',  np.arccos(np.dot(inject_angle_rotate, inject_normal_rotate)))
# ax.quiver(0, 0, 0, -normal[0], -normal[1], -normal[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'blue')
ax.quiver(0, 0, 0, -normal[0], -normal[1], -normal[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'blue')
# ax.quiver(normal[0], normal[1], normal[2], 0, 0, 0, length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'blue')
ax.quiver(0, 0, 0, vel[0], vel[1], vel[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')
# ax.quiver(0, 0, 0, inject_normal_rotate[0], inject_normal_rotate[1], inject_normal_rotate[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'green')
# ax.quiver(0, 0, 0, inject_angle_rotate[0], inject_angle_rotate[1], inject_angle_rotate[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')
# plt.show()
ax.view_init(elev=0, azim=-90, roll=0)
fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式