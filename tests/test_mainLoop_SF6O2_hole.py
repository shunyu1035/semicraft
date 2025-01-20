import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import pyvista as pv
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import src.postProcess as PostProcess
import src.particleGenerator as particleGenerator
import src.operations.sputterYield as sputterYield
from src.mainLoop_int_yield import mainLoop


if __name__ == "__main__":
    # vertical mask
    film = np.zeros((100, 100, 160, 11), dtype=np.int32)

    bottom = 120
    height = 140

    density = 20

    sphere = np.ones((100, 100, 160), dtype=bool)

    radius = 30

    center = 50
    for i in range(sphere.shape[0]):
        for j in range(sphere.shape[1]):
            if np.abs(i-center)*np.abs(i-center) + np.abs(j-center)*np.abs(j-center) < radius*radius:
                sphere[i, j, bottom:height] = 0

    film[sphere, -1] = density
    film[:, :, height:, :] = 0
    film[:, :, 0:bottom, 0] = density # bottom
    film[:, :, 0:bottom, 1:] = 0 # bottom

    etchfilm = film
    center = film.shape[1]/2

    yield_hist = sputterYield.sputterYield_ion

    logname = './logfiles/sf6o2ion_1206'
    etchingPoint = np.array([center, center, bottom-30])
    depoPoint = np.array([center, center, bottom-30])
    density = 10
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
    offset_distence = 0.8
    reaction_type=False
    celllength=1e-5
    kdtreeN=2
    filmKDTree=np.array([[2, 0, -1], [3, 0, -1]]) # 1 for depo -1 for etching
    # filmKDTree=np.array([[2, 1], [3, 1]])
    weightDepo = 10
    weightEtching = 10
    tstep=1e-5
    substrateTop=bottom




    particle_list = [[int(1e6), 0, 'maxwell', 50], [int(1e6), 1, 'maxwell', 60], [int(1e5), 2, 'updown', 60]]
    # particle_list = [[int(1e6), 0, 'maxwell', 50], [int(1e6), 1, 'maxwell', 60]]
    # particle_list = [[int(1e6), 0, 'maxwell', 50]]
    vel_matrix = particleGenerator.vel_generator(particle_list)

    parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 10, 50, 0]])


    # print(config.cellSizeX)
    # PostProcess.PostProcess(etchfilm, colors=['dimgray', 'yellow', 'cyan'])
    testMain = mainLoop(etchingPoint,depoPoint,
                        density, center_with_direction, 
                        range3D, InOrOut, yield_hist,
                        maskTop, maskBottom, maskStep, maskCenter,backup, 
                        mirrorGap,offset_distence,
                        reaction_type,
                        celllength, kdtreeN, filmKDTree,weightDepo,weightEtching, tstep,
                        substrateTop, logname)

    testMain.input(etchfilm, parcel,'etching', vel_matrix, 0)

    # print(testMain.cellSizeX)
    testMain.runEtch(int(1e4), int(1e5), int(2e7))

    # labels = ['Si', 'SiF1', 'SiF2', 'SiF3', 'SiO', 'SiO2', 'SiOF', 'SiOF2', 'SiO2F', 'SiO2F2', 'mask']
    # color_names = ['dimgray', 'blue', 'red', 'green', 'yellow', 'brown', 'magenta', 'orange', 'purple', 'pink', 'cyan', 'black', 'white', 'gray']
    # PostProcess.PostProcess_multiLayer(etchfilm, colors=color_names, labels=labels)
