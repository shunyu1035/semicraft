# -*- coding: utf-8 -*-

import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.operations.surface as surface
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv

plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


film = np.zeros((100, 100, 150))

bottom = 78
# film[:, :, 0:bottom] = 10 # bottom

height = 80
# left_side = 75
# right_side = 75
# film[:, left_side+6:200-left_side-6, 0:height] = 10
film[45:55, 45:55, 0:height] = 10
# film[:, :55, 0:height] = 10
# film[45:, :, 0:height] = 10
# film[:55, :, 0:height] = 10

film[:, :, 0:bottom] = 0 # bottom
film[:, :, height:] = 0 # bottom

etchfilm = np.zeros((100, 100, 150, 2))
etchfilm[:, :, :, 0] = film
# etchfilm[:, :, :, 1] = film

center = 50


yield_hist = np.array([[1.0, 1.01, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0, \
                        0.08, 0.28,0.65,  1.07, 1.5, 1.4, 1.2, 1.05, 1.01, 1.0 ], \
                        [  0,  5,   10,   20,   30,   40,   50,   60,   70,   80, 90, \
                          100, 110, 120, 130, 140, 150, 160, 170, 175, 180]])
yield_hist[1] *= np.pi/180


center_with_direction=np.array([[int(etchfilm.shape[0]/2),int(etchfilm.shape[1]/2),150]])
range3D=np.array([[0, etchfilm.shape[0], 0, etchfilm.shape[1], 0, etchfilm.shape[2]]])
InOrOut=[1]
# yield_hist=np.array([None])
yield_hist = yield_hist
maskTop=40, 
maskBottom=98, 
maskStep=10, 
maskCenter=[int(etchfilm.shape[0]/2), int(etchfilm.shape[1]/2)]
backup=False
mirrorGap=5
offset_distence = 0.5
reaction_type=False
param = [1.6, -0.7]
n=1
celllength=1e-5
kdtreeN=2
tstep=1e-5
density = 10

test = surface.surface_normal(center_with_direction, range3D, InOrOut,celllength, tstep, yield_hist,\
                        maskTop, maskBottom, maskStep, maskCenter, backup, density, mirrorGap, offset_distence)


def update_surface_mirror(surface_etching):
    cellSizeX = surface_etching.shape[0]
    cellSizeY = surface_etching.shape[1]
    cellSizeZ = surface_etching.shape[2]
    mirrorGap = 5
    surface_etching_mirror = np.zeros((cellSizeX+int(mirrorGap*2), cellSizeY+int(mirrorGap*2), cellSizeZ))

    surface_etching_mirror[mirrorGap:mirrorGap+cellSizeX, mirrorGap:mirrorGap+cellSizeY, :] = surface_etching
    surface_etching_mirror[:mirrorGap, mirrorGap:mirrorGap+cellSizeY, :] = surface_etching[-mirrorGap:, :, :]
    surface_etching_mirror[-mirrorGap:, mirrorGap:mirrorGap+cellSizeY, :] = surface_etching[:mirrorGap, :, :]
    surface_etching_mirror[mirrorGap:mirrorGap+cellSizeX, :mirrorGap, :] = surface_etching[:, -mirrorGap:, :]
    surface_etching_mirror[mirrorGap:mirrorGap+cellSizeX:, -mirrorGap:, :] = surface_etching[:, :mirrorGap, :]
    surface_etching_mirror[:mirrorGap, :mirrorGap, :] = surface_etching[-mirrorGap:, -mirrorGap:, :]
    surface_etching_mirror[:mirrorGap, -mirrorGap:, :] = surface_etching[-mirrorGap:, :mirrorGap, :]
    surface_etching_mirror[-mirrorGap:, :mirrorGap, :] = surface_etching[:mirrorGap, -mirrorGap:, :]
    surface_etching_mirror[-mirrorGap:, -mirrorGap:, :] = surface_etching[:mirrorGap, :mirrorGap, :]

    return surface_etching_mirror

sumFilm = film
sumFilm_mirror = update_surface_mirror(sumFilm)
pos1e4_data = test.get_pointcloud(sumFilm_mirror)
point_cloud = pv.PolyData(pos1e4_data[:, 3:6])
vectors = pos1e4_data[:, :3]

point_cloud['vectors'] = vectors
arrows = point_cloud.glyph(
    orient='vectors',
    scale=10000,
    factor=3,
)

# Display the arrowscyan


plotter = pv.Plotter()
plotter.add_mesh(point_cloud, color='cyan', point_size=5.0, render_points_as_spheres=True)
# plotter.add_mesh(sphere, show_edges=True, opacity=0.5, color="w")
plotter.add_mesh(arrows, color='lightblue')
# # plotter.add_point_labels([point_cloud.center,], ['Center',],
# #                          point_color='yellow', point_size=20)
plotter.show_grid()
plotter.show()