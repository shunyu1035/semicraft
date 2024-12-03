import numpy as np
import pyvista as pv

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