import sys
sys.path.append("./")  # 确保根目录在 sys.path 中
import src.simulator as simulator
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pyvista as pv
import torch

# def slide2D_fractionZ(film, start, end, direction, fraction, value):
#     if fraction == '+':
#         if direction == 'y':
#             slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
#             fraction = np.abs(int(slit[0]-slit[1]))
#             print('y', slit)
#             print('fraction', fraction)
#             for i in range(np.abs(end[1] - start[1])):
#                 if end[1] > start[1]:
#                     film[start[0]:end[0], start[1] + i, start[2]:start[2] + int(slit[i])] = value
#                     for j in range(fraction):
#                         film[start[0]:end[0], start[1] + i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
#                 elif end[1] < start[1]:
#                     film[start[0]:end[0], start[1] - i, start[2]:start[2] + int(slit[i])] = value
#                     for j in range(fraction):
#                         film[start[0]:end[0], start[1] - i,start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
#         elif direction == 'x':
#             slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
#             fraction = np.abs(int(slit[0]-slit[1]))
#             print('x', slit)
#             print('fraction', fraction)
#             for i in range(np.abs(end[2] - start[2])):
#                 if end[0] > start[0]:
#                     film[start[0] + i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
#                     for j in range(fraction):
#                         film[start[0] + i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
#                 elif end[0] < start[0]:
#                     film[start[0] - i, start[1]:end[1], start[2]:start[2] + int(slit[i])] = value
#                     for j in range(fraction):
#                         film[start[0] - i, start[1]:end[1], start[2]+int(slit[i])+j] = 1/(fraction+1)*(fraction-j)
#     elif fraction == '-':
#         if direction == 'y':
#             slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[1] - start[1]))
#             fraction = np.abs(int(slit[0]-slit[1]))
#             print('y', slit)
#             print('fraction', fraction)
#             for i in range(np.abs(end[1] - start[1])):
#                 if end[1] > start[1]:
#                     film[start[0]:end[0], start[1] + i, start[2] - int(slit[i]):start[2]+1] = value
#                     for j in range(fraction):
#                         film[start[0]:end[0], start[1] + i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
#                 elif end[1] < start[1]:
#                     film[start[0]:end[0], start[1] - i, start[2] - int(slit[i]):start[2]+1] = value
#                     for j in range(fraction):
#                         film[start[0]:end[0], start[1] - i,start[2]-int(slit[i])-j] = 1/(fraction+1)*(fraction-j)
#         elif direction == 'x':
#             slit = np.linspace(0, np.abs(end[2] - start[2]), np.abs(end[0] - start[0]))
#             fraction = np.abs(int(slit[0]-slit[1]))
#             print('x', slit)
#             print('fraction', fraction)
#             for i in range(np.abs(end[2] - start[2])):
#                 if end[0] > start[0]:
#                     film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
#                     for j in range(fraction):
#                         film[start[0] + i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
#                 elif end[0] < start[0]:
#                     film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]+1] = value
#                     for j in range(fraction):
#                         film[start[0] - i, start[1]:end[1], start[2] - int(slit[i]):start[2]] = 1/(fraction+1)*(fraction-j)
#     return film

# film = np.zeros((70, 200, 150))

# bottom = 10
# # film[:, :, 0:bottom] = 10 # bottom

# height = 80
# left_side = 71
# right_side = 71

# slit = 8
# film[:, left_side+slit:200-right_side-slit, 0:height] = 10

# left_side_gap = 19
# right_side_gap = 181
# film[:, :left_side_gap, 0:height] = 10
# film[:, right_side_gap:, 0:height] = 10

# film = slide2D_fractionZ(film=film, start=[0, left_side, bottom], end=[70, left_side+slit, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, 200-right_side-1, bottom], end=[70, 200-right_side-slit-1, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, left_side_gap+slit-1, bottom], end=[70, left_side_gap-1, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, right_side_gap-slit, bottom], end=[70, right_side_gap, height], direction='y', fraction='+', value=10)

# # film[:, 80:121, 0:31] = 10

# film[:, :, 0:bottom] = 10 # bottom
# film[:, :, height:] = 0 # bottom

yield_hist = np.array([[1.0, 1.01, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0, \
                        0.08, 0.28,0.65,  1.07, 1.5, 1.4, 1.2, 1.05, 1.01, 1.0 ], \
                        [  0,  5,   10,   20,   30,   40,   50,   60,   70,   80, 90, \
                        100, 110, 120, 130, 140, 150, 160, 170, 175, 180]])
yield_hist[1] *= np.pi/180

# etchfilm = np.zeros((70, 200, 150, 2))
# etchfilm[:, :, :, 0] = film
# # etchfilm[:, :, :, 1] = film

# center = 100


film = np.zeros((70, 100, 150))

bottom = 40
# film[:, :, 0:bottom] = 10 # bottom

height = 80
# left_side = 75
# right_side = 75
# film[:, left_side+6:200-left_side-6, 0:height] = 10
film[:, :45, 0:height] = 10
film[:, 55:, 0:height] = 10

# film = slide2D_fractionZ(film=film, start=[0, left_side, bottom], end=[70, left_side+6, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, 200-left_side-1, bottom], end=[70, 200-left_side-6-1, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, 19+6-1, bottom], end=[70, 19-1, height], direction='y', fraction='+', value=10)
# film = slide2D_fractionZ(film=film, start=[0, 181-6, bottom], end=[70, 181, height], direction='y', fraction='+', value=10)

# film[:, 80:121, 0:31] = 10

film[:, :, 0:bottom] = 10 # bottom
film[:, :, height:] = 0 # bottom

etchfilm = np.zeros((70, 100, 150, 2))
etchfilm[:, :, :, 0] = film
# etchfilm[:, :, :, 1] = film

center = 50
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
offset_distence = 0.8
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
                    mirrorGap, offset_distence,
                    reaction_type, param,n,
                    celllength, kdtreeN, filmKDTree,weightDepo,weightEtching, tstep,
                    substrateTop,posGeneratorType, logname)


# cicle = 100
# celllength=1e-5
# parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 0.2, 0]])
# step1 = testEtch.inputParticle(etchfilm, parcel, 'depo', 'maxwell', 0, 0, int(5e3), int(1e6), int(1e5),2, 4, 100)

N = int(1e7)
velosity_matrix = np.zeros((N, 3))

velosity_matrix[:, 0] = 0
velosity_matrix[:, 1] = 0
velosity_matrix[:, 2] = -1 

cicle = 100
celllength=1e-5
parcel = np.array([[95*celllength, 95*celllength, 159*celllength, 0, 0, 1, 95, 95, 159, 0.2, 0]])
step1 = testEtch.inputParticle(etchfilm, parcel,'etching', 'input',velosity_matrix, 1, int(1e4), int(1e5), int(4e4),2, 10, 100)

geom = pv.Box()

substrute = torch.Tensor(np.logical_and(etchfilm[:, :, :,0]>0, etchfilm[:, :, :,0]<1)).to_sparse()
substrute = substrute.indices().numpy().T

# cyan = torch.Tensor(np.logical_and(etchfilm[:, :, :,0]!=0, film[:, :, :]!=10)).to_sparse()
cyan = torch.Tensor(etchfilm[:, :, :,0]>=9).to_sparse()
cyan = cyan.indices().numpy().T

submesh = pv.PolyData(substrute)
submesh["radius"] = np.ones(substrute.shape[0])*0.5

cyanmesh = pv.PolyData(cyan)
cyanmesh["radius"] = np.ones(cyan.shape[0])*0.5
# Progress bar is a new feature on master branch
cyanglyphed = cyanmesh.glyph(scale="radius", geom=geom) # progress_bar=True)
subglyphed = submesh.glyph(scale="radius", geom=geom) # progress_bar=True)
p = pv.Plotter()
# p.add_mesh(subglyphed, color='gray')
p.add_mesh(cyanglyphed, color='cyan')
p.enable_eye_dome_lighting()
p.show()