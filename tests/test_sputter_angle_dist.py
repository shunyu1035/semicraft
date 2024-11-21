# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.sputter_angle_dist as sp_angle
import numpy as np
import matplotlib.pyplot as plt
import os

plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

theta, phi = np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi/2, np.pi/2, 100)
THETA, PHI = np.meshgrid(theta, phi)
# R = np.cos(PHI**2)
inject_angle = np.pi/12

R = sp_angle.emission_angle( THETA, PHI, theta1 = inject_angle, Eth=26.8, E = 50)
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
# Z = np.abs(R * np.cos(THETA))
Z = R * np.cos(THETA)
renormal = Z < 0
Z[renormal] = 0
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    linewidth=0, antialiased=False, alpha=0.2)

ax.quiver(0, 0, 0, -1, 0, -np.tan(np.pi/2-inject_angle), length=1,arrow_length_ratio = 0.1, pivot = 'tip',normalize=True, colors = 'red')
# plt.show()
ax.view_init(elev=0, azim=-90, roll=0)
fig.show()
input("按任意键退出交互模式...")
plt.ioff()  # 关闭交互模式