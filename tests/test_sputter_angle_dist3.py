# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.sputter_angle_dist as sp_angle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as Rotate


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

plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

Eth = 26.8
resolu = 100

    # print(i)
# E = np.random.uniform(40, 60)
# theta1 = np.random.uniform(np.pi/100, np.pi/2)

normal = np.array([0, 0.5, 1])
normal = np.divide(normal, np.linalg.norm(normal, axis=-1))

normal_t = np.arccos(np.dot(normal, np.array([0, 0, 1])))
normal_p = np.arctan2(normal[0], normal[1])

vel = np.array([-1, 0, -1])
vel = np.divide(vel, np.linalg.norm(vel, axis=-1))

inject_angle = np.arccos(np.dot(normal, vel))

print('angle', inject_angle)
print('normal_t', normal_t/np.pi*180)
# inject_angle = np.pi/12
E = 50

# r = Rotate.from_rotvec([0, normal_t, 0])
# r = Rotate.from_rotvec([-normal_t, 0, 0])
N = 200

for i in range(N):
    R, theta, phi = sp_angle.phi_theta_dist(resolu, inject_angle, Eth, E)
    get_theta, get_phi = sp_angle.accept_reject_phi_theta(R, theta, phi)

    # X = R * np.sin(THETA) * np.cos(PHI)
    # Y = R * np.sin(THETA) * np.sin(PHI)
    # # Z = np.abs(R * np.cos(THETA))
    # Z = R * np.cos(THETA)
    phi0 = np.arctan2(vel[1], vel[0]) + np.pi
    # phi0 = 0

    velX = np.sin(get_theta)*np.cos(get_phi+phi0)
    velY = np.sin(get_theta)*np.sin(get_phi+phi0)
    velZ = np.cos(get_theta)

    vel_reflect = np.array([velX, velY, velZ])
    vel_reflect_norm = np.linalg.norm(vel_reflect)

    vel_reflect = vel_reflect @ rotationRX(normal_t)
    # vel_reflect = rotate_vector_spherical(vel_reflect, normal_t, normal_p)
    # velX = np.sin(get_theta+normal_t)*np.cos(get_phi+phi0+normal_p)
    # velY = np.sin(get_theta+normal_t)*np.sin(get_phi+phi0+normal_p)
    # velZ = np.cos(get_theta+normal_t)

    # velX = velX*np.sin(normal_t)*np.cos(normal_p)
    # velY = velY*np.sin(normal_t)*np.sin(normal_p)
    # velZ = velZ*np.cos(normal_t)


    # print(vel_reflect)
    ax.quiver(0, 0, 0, vel_reflect[0], vel_reflect[1], vel_reflect[2], arrow_length_ratio = 0.1, pivot = 'tail', colors = 'black')


# print('angle between:',  np.arccos(np.dot(inject_angle_rotate, inject_normal_rotate)))
# normal2 = rotate_vector_spherical(normal, normal_t, normal_p)
# ax.quiver(0, 0, 0, normal2[0], normal2[1], normal2[2], length=3,arrow_length_ratio = 0.1, pivot = 'tail',normalize=True, colors = 'green')
ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], length=2,arrow_length_ratio = 0.1, pivot = 'tail',normalize=True, colors = 'blue')
ax.quiver(0, 0, 0, vel[0], vel[1], vel[2], length=2,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')
# plt.show()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=0, azim=-90, roll=0)
fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式



