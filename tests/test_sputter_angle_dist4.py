# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.sputter_angle_dist as sp_angle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as Rotate


plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def SpecularReflect(vel, normal):
    return vel - 2*vel@normal*normal

def DiffusionReflect(vel, normal):
    Ut = vel - vel@normal*normal
    tw1 = Ut/np.linalg.norm(Ut)
    tw2 = np.cross(tw1, normal)
    # U = np.sqrt(kB*T/particleMass[i])*(np.random.randn()*tw1 + np.random.randn()*tw2 - np.sqrt(-2*np.log((1-np.random.rand())))*normal)
    U = np.random.randn()*tw1 + np.random.randn()*tw2  + np.sqrt(-2*np.log((1-np.random.rand())))*normal
    UN = U / np.linalg.norm(U)
        # UN[i] = U
    return UN

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

Eth = 26.8
resolu = 100


normal = np.array([0, 0.5, 1])
normal = np.divide(normal, np.linalg.norm(normal, axis=-1))

vel = np.array([1, 0, -1])
vel = np.divide(vel, np.linalg.norm(vel, axis=-1))

inject_angle = np.arccos(np.dot(normal, vel))

print('angle', inject_angle)
# inject_angle = np.pi/12
E = 50
N = 200

for i in range(N):

    vel_reflect = DiffusionReflect(vel, normal)
    # vel_reflect = SpecularReflect(vel, normal)
    print(vel_reflect)
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



