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