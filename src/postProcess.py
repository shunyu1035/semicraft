import numpy as np
import pyvista as pv
from numba import jit, prange


@jit(nopython=True)
def layerLoop(film):
    layer = np.zeros((film.shape[0],film.shape[1],film.shape[2]))
    for i in range(film.shape[0]):
        for j in range(film.shape[1]):
            for k in range(film.shape[2]):
                for c in range(film.shape[3]):
                    if np.sum(film[i, j, k]) > 0:
                        layer[i, j, k] = np.argmax(film[i, j, k]) + 1 
    return layer

def PostProcess_multiLayer(film, colors=['dimgray', 'yellow', 'cyan'], labels=['Si', 'SiClx', 'Mask']):
    if film.shape[3] > len(colors) or film.shape[3] > len(labels):
        print('error: please set colors or labels')
        return 0 
    geom = pv.Box()
    p = pv.Plotter()

    layer =layerLoop(film)

    for ci in range(film.shape[3]):
        layerCube = np.argwhere(layer == ci+1)
        if layerCube.size != 0:
            layermesh = pv.PolyData(layerCube)
            layermesh["radius"] = np.ones(layerCube.shape[0])*0.5
            layerglyphed = layermesh.glyph(scale="radius", geom=geom, orient=False) # progress_bar=True)
            p.add_mesh(layerglyphed, color=colors[ci], label=labels[ci])
    p.enable_eye_dome_lighting()
    p.add_legend()
    p.show()

def PostProcess(film, colors=['dimgray', 'yellow', 'cyan']):
    if film.shape[3] > len(colors):
        print('error: please set colors')
        return 0 
    geom = pv.Box()
    p = pv.Plotter()

    for i in range(film.shape[3]):
        layer = np.argwhere(film[:, :, :, i] != 0)
        if layer.size != 0:
            layermesh = pv.PolyData(layer)
            layermesh["radius"] = np.ones(layer.shape[0])*0.5
            layerglyphed = layermesh.glyph(scale="radius", geom=geom, orient=False) # progress_bar=True)
            p.add_mesh(layerglyphed, color=colors[i])
    p.enable_eye_dome_lighting()
    p.show()



def surface_vector(plane):
    point_cloud = pv.PolyData(plane[:, 3:6])
    vectors = plane[:, :3]

    point_cloud['vectors'] = vectors
    arrows = point_cloud.glyph(
        orient='vectors',
        scale=10000,
        factor=3,
    )
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='cyan', point_size=5.0, render_points_as_spheres=True)
    plotter.add_mesh(arrows, color='lightblue')
    plotter.show_grid()
    plotter.show()

@jit(nopython=True)
def transfer_to_plane(normal_array):
    plane = np.zeros((normal_array.shape[0]*normal_array.shape[1]*normal_array.shape[2], 6))
    count = 0
    for i in range(normal_array.shape[0]):
        for j in range(normal_array.shape[1]):
            for k in range(normal_array.shape[2]):
                if np.sum(normal_array[i, j, k]) != 0:
                    plane[count, :3] = normal_array[i, j, k]
                    plane[count, 3:6] = i, j, k
                    count += 1
    return plane[:count]


def surface_vector_normal_array(normal_array):
    plane = transfer_to_plane(normal_array)
    point_cloud = pv.PolyData(plane[:, 3:6])
    vectors = plane[:, :3]

    point_cloud['vectors'] = vectors
    arrows = point_cloud.glyph(
        orient='vectors',
        scale=10000,
        factor=3,
    )
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='cyan', point_size=5.0, render_points_as_spheres=True)
    plotter.add_mesh(arrows, color='lightblue')
    plotter.show_grid()
    plotter.show()