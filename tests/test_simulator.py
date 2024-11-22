import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.simulator as simulator
import numpy as np
import matplotlib.pyplot as plt
import os


def slide2D_fractionZ(film, start, end, direction, fraction, value):
    if fraction == '+':
        if direction == 'y':
            slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
            fraction = np.abs(int(slit[0]-slit[1]))
            print('y', slit)
            print('fraction', fraction)
            for i in range(np.abs(end[1] - start[1])):
                if end[1] > start[1]:
                    film[start[0]:end[0], start[1] + i, start[2]:start[2] + int(slit[i])] = value
                    for j in range(fraction):
                        film[start[0]:end[0], start[1] + i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
                elif end[1] < start[1]:
                    film[start[0]:end[0], start[1] - i, start[2]:start[2] + int(slit[i])] = value
                    for j in range(fraction):
                        film[start[0]:end[0], start[1] - i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
        elif direction == 'x':
            slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
            fraction = np.abs(int(slit[0]-slit[1]))
            print('x', slit)
            print('fraction', fraction)
            for i in range(np.abs(end[2] - start[2])):
                if end[0] > start[0]:
                    film[start[0] + i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
                    for j in range(fraction):
                        film[start[0] + i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
                elif end[0] < start[0]:
                    film[start[0] - i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
                    for j in range(fraction):
                        film[start[0] - i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
    elif fraction == '-':
        if direction == 'y':
            slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
            fraction = np.abs(int(slit[0]-slit[1]))
            print('y', slit)
            print('fraction', fraction)
            for i in range(np.abs(end[1] - start[1])):
                if end[1] > start[1]:
                    film[start[0]:end[0], start[1] + i, start[2] - int(slit[i]):start[2]+1] = value
                    for j in range(fraction):
                        film[start[0]:end[0], start[1] + i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
                elif end[1] < start[1]:
                    film[start[0]:end[0], start[1] - i, start[2] - int(slit[i]):start[2]+1] = value
                    for j in range(fraction):
                        film[start[0]:end[0], start[1] - i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
        elif direction == 'x':
            slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
            fraction = np.abs(int(slit[0]-slit[1]))
            print('x', slit)
            print('fraction', fraction)
            for i in range(np.abs(end[2] - start[2])):
                if end[0] > start[0]:
                    film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
                    for j in range(fraction):
                        film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
                elif end[0] < start[0]:
                    film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
                    for j in range(fraction):
                        film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
    return film

film = np.zeros((70, 200, 150))

bottom = 10
# film[:, :, 0:bottom] = 10 # bottom

height = 80
left_side = 71
right_side = 71

slit = 8
film[:, left_side+slit:200-right_side-slit, 0:height] = 10

left_side_gap = 19
right_side_gap = 181
film[:, :left_side_gap, 0:height] = 10
film[:, right_side_gap:, 0:height] = 10

film = slide2D_fractionZ(film=film, start=[0, left_side, bottom], end=[70, left_side+slit, height], direction='y', fraction='+', value=10)
film = slide2D_fractionZ(film=film, start=[0, 200-right_side-1, bottom], end=[70, 200-right_side-slit-1, height], direction='y', fraction='+', value=10)
film = slide2D_fractionZ(film=film, start=[0, left_side_gap+slit-1, bottom], end=[70, left_side_gap-1, height], direction='y', fraction='+', value=10)
film = slide2D_fractionZ(film=film, start=[0, right_side_gap-slit, bottom], end=[70, right_side_gap, height], direction='y', fraction='+', value=10)

# film[:, 80:121, 0:31] = 10

film[:, :, 0:bottom] = 10 # bottom
film[:, :, height:] = 0 # bottom

yield_hist = np.array([[1.0, 1.01, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0, \
                        0.08, 0.28,0.65,  1.07, 1.5, 1.4, 1.2, 1.05, 1.01, 1.0 ], \
                        [  0,  5,   10,   20,   30,   40,   50,   60,   70,   80, 90, \
                        100, 110, 120, 130, 140, 150, 160, 170, 175, 180]])
yield_hist[1] *= np.pi/180

etchfilm = np.zeros((70, 200, 150, 2))
etchfilm[:, :, :, 0] = film
# etchfilm[:, :, :, 1] = film

center = 100



# ----------------------------------------------------------------------------------------------

logname = 'Multi_species_benchmark_1021_hole_ratio01'
inputMethod='bunch'
etchingPoint = np.array([center, center, 125])
depoPoint = np.array([center, center, 125])
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
reaction_type=False
param = [1.6, -0.7]
n=1
celllength=1e-5
kdtreeN=5
filmKDTree=np.array([[2, 0, 1], [3, 0, -1]]) # 1 for depo -1 for etching
# filmKDTree=np.array([[2, 1], [3, 1]])
weightDepo=0.2
weightEtching = 0.2
tstep=1e-5
substrateTop=80
posGeneratorType='top'
testEtch = simulator.etching(
                    inputMethod,
                    etchingPoint,depoPoint,
                    density, center_with_direction, 
                    range3D, InOrOut, yield_hist,
                    maskTop, maskBottom, maskStep, maskCenter,backup, 
                    mirrorGap,
                    reaction_type, param,n,
                    celllength, kdtreeN, filmKDTree,weightDepo,weightEtching, tstep,
                    substrateTop,posGeneratorType, logname)


cicle = 100
celllength=1e-5
parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 0.2, 0]])
step1 = testEtch.inputParticle(etchfilm, parcel, 'depo', 'maxwell', 0, 0, int(5e3), int(1e6), int(1e5),2, 4, 100)