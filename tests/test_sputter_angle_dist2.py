# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.sputter_angle_dist as sp_angle
import numpy as np
import matplotlib.pyplot as plt
import os

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



normal_t = np.array([1, 0, 3**0.5])
normal_t = np.divide(normal_t, np.linalg.norm(normal_t, axis=-1))
normal_theta = np.arccos(np.dot(normal_t, np.array([0, 0, 1])))
print('normalt',normal_theta/np.pi *180)

normal = np.array([0, 0, 1])

phi = np.arccos(np.dot(normal, np.array([0, 1, 1])))
phi = np.pi/6

normal_rotate = np.dot(rotationRY(phi), normal)
normal_rotate = np.dot(rotationRZ(phi), normal_rotate)

theta = np.arccos(np.dot(normal_rotate, np.array([0, 0, 1])))
print(theta/np.pi *180)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

ax.quiver(0, 0, 0, -normal[0], -normal[1], -normal[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'blue')
ax.quiver(0, 0, 0, -normal_t[0], -normal_t[1], -normal_t[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'green')
ax.quiver(0, 0, 0, -normal_rotate[0], -normal_rotate[1], -normal_rotate[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')


# plt.show()
ax.view_init(elev=0, azim=-90, roll=0)
fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式


'''

theta, phi = np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi/2, np.pi/2, 100)
THETA, PHI = np.meshgrid(theta, phi)
# R = np.cos(PHI**2)
inject_angle_array = np.array([1, -2, -1])
inject_angle_array = np.divide(inject_angle_array, np.linalg.norm(inject_angle_array, axis=-1))

normal = np.array([0, 0, 1])
normal = np.divide(normal, np.linalg.norm(normal, axis=-1))

inject_angle = np.arccos(np.dot(normal, inject_angle_array))

print('angle', inject_angle)

phi0 = np.arctan2(inject_angle_array[1], inject_angle_array[0]) + np.pi
# phi0 = np.arccos(np.dot(inject_angle_array, np.array([0, 1, 0])))
# theta0 = np.arccos(np.dot(normal, np.array([0, 0, 1])))
theta0 = 0

print('theta0', theta0)
print('phi0', phi0)

# phi0 = 0
inject_angle =np.pi/12

R = sp_angle.emission_angle( THETA, PHI, theta1 = inject_angle, Eth=26.8, E = 50)

X = np.zeros_like(R)
Y = np.zeros_like(R)
Z = np.zeros_like(R)


# normal = np.array([0, 0, 1])
# normal = np.divide(normal, np.linalg.norm(normal, axis=-1))

# phi0 = np.pi
# theta0 = np.pi

X = R * np.sin(THETA) * np.cos(PHI+phi0)
Y = R * np.sin(THETA) * np.sin(PHI+phi0)
# Z = np.abs(R * np.cos(THETA))
Z = R * np.cos(THETA)
# renormal = Z < 0
# Z[renormal] = 0
    


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    linewidth=0, antialiased=False, alpha=0.2)

# inject_angle_array = np.array([-1, 0, -np.tan(np.pi/2-inject_angle)])
# inject_angle_array = np.divide(inject_angle_array, np.linalg.norm(inject_angle_array, axis=-1))
# inject_angle_rotate = np.dot(inject_angle_array, rotation_matrix(normal_t, normal_p))
# inject_normal_rotate = np.dot(np.array([0, 0, -1]), rotation_matrix(normal_t, normal_p))

# print('angle between:',  np.arccos(np.dot(inject_angle_rotate, inject_normal_rotate)))
ax.quiver(0, 0, 0, -normal[0], -normal[1], -normal[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'blue')
# ax.quiver(0, 0, 0, inject_normal_rotate[0], inject_normal_rotate[1], inject_normal_rotate[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'green')
ax.quiver(0, 0, 0, inject_angle_array[0], inject_angle_array[1], inject_angle_array[2], length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')
# plt.show()
ax.view_init(elev=0, azim=-90, roll=0)
fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式

'''