import sys
sys.path.append("./")  # 确保根目录在 sys.path 中

import src.postProcess as PostProcess
import src.particleGenerator as particleGenerator
import src.operations.sputterYield as sputterYield
from src.config import sputter_yield
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from src.config import sputter_yield
from src.mainLoop import mainLoop
import numpy as np

if __name__ == "__main__":
    film = np.zeros((20, 100, 140, 3))

    bottom = 100
    height = 104

    density = 10

    center = 50

    film[:, :45, bottom:height, 2] = density
    film[:, 55:, bottom:height, 2] = density
    # film[:, :, 0:bottom, :] = 0
    film[:, :, 0:bottom, 0] = density # bottom

    etchfilm = film
    center = film.shape[1]/2

    yield_hist = sputterYield.sputterYield_Func(sputter_yield[0], sputter_yield[1], sputter_yield[2])

    logname = './logfiles/simulator_ver1_1203'
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
    weightDepo=0.2
    weightEtching = 0.2
    tstep=1e-5
    substrateTop=bottom




    # particle_list = [[int(1e6), 0, 'maxwell', 50], [int(1e6), 1, 'undown', 60]]
    particle_list = [[int(1e6), 0, 'maxwell', 50]]
    vel_matrix = particleGenerator.vel_generator(particle_list)

    parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 0.2, 50, 0]])


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
    
    testMain.input(etchfilm, parcel,'etching', vel_matrix, 2)

    print(testMain.cellSizeX)
    testMain.runEtch(int(1e4), int(1e5), int(1e5))
    PostProcess.PostProcess(etchfilm, colors=['dimgray', 'yellow', 'cyan'])
