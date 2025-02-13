import sys
sys.path.append("../")  # 确保根目录在 sys.path 中
from PIL import Image
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import src.postProcess as PostProcess
import src.particleGenerator as particleGenerator

import SimProfile

react_table_equation = np.array([
    [
        [-1, 1, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [-1, 0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, -1]
    ]
])


react_type_table = np.array([[1, 1, 1, 4, 0], # 1: chlorination  # 0: no reaction  # 4: Themal etch
                            [2, 2, 2, 2, 2], # 2 for physics and chemical sputtering
                            [3, 3, 3, 3, 3]])

react_prob_chemical = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
react_yield_p0 = np.array([0.30, 0.30, 0.30, 0.30, 0.30])


rn_coeffcients = np.array([[0.9423, 0.9434, 2.742, 3.026],
                            [0.9620, 0.9608, 2.542, 3.720],
                            [0.9458, 0.9445, 2.551, 3.735],
                            [1.046, 1.046, 2.686, 4.301]])


# 读取图像为灰度模式
image = Image.open("./hard_mask_KLA3.jpg").convert("L")
# 转换为 NumPy 数组
HardMasK = np.array(image)

# Hard mask
film = np.zeros((20, HardMasK.shape[1], HardMasK.shape[0], 5), dtype=np.int32)
density = 20
for i in range(HardMasK.shape[1]):
    for j in range(HardMasK.shape[0]):
        if HardMasK[j, i] < 80: # Si
            film[:, i, -j, 0] = density
        if HardMasK[j, i] >= 80 and HardMasK[j, i] < 120: # HM
            film[:, i, -j, -1] = density

film[:, :, :450, -1] = 0
# film[:, :, :60, -1] = 0
film[:, :, :2, -1] = density

Cell_dtype = np.dtype([
    ('id', np.int32),
    ('index', np.int32, (3,)),
    ('film', np.int32, (5,)),
    ('normal', np.float64, (3,))
], align=True)  # 添加 align=True

cell = np.zeros((20, HardMasK.shape[1], HardMasK.shape[0]), dtype=Cell_dtype)

cell['film'] = film

def scanZ_numpy_bool(film):
    # 初始化一个全零的表面数组
    surface_sparse_depo = np.zeros_like(film, dtype=np.bool_)

    # depo 条件
    current_plane = film != 0

    # 创建邻居布尔索引数组
    neighbors = np.zeros_like(film, dtype=bool)

    # 获取周围邻居的布尔索引
    neighbors[1:, :, :] |= film[:-1, :, :] == 0  # 上面
    neighbors[:-1, :, :] |= film[1:, :, :] == 0  # 下面
    neighbors[:, 1:, :] |= film[:, :-1, :] == 0  # 左边
    neighbors[:, :-1, :] |= film[:, 1:, :] == 0  # 右边
    neighbors[:, :, 1:] |= film[:, :, :-1] == 0  # 前面
    neighbors[:, :, :-1] |= film[:, :, 1:] == 0  # 后面

    # 获取满足条件的索引
    condition_depo = current_plane & neighbors

    # 更新表面稀疏张量
    surface_sparse_depo[condition_depo] = True

    return surface_sparse_depo

def scanZ_vacuum_numpy_bool(film):
    # 初始化一个全零的表面数组
    surface_sparse_depo = np.zeros_like(film, dtype=np.bool_)

    # depo 条件
    current_plane = film == 0

    # 创建邻居布尔索引数组
    neighbors = np.zeros_like(film, dtype=bool)

    # 获取周围邻居的布尔索引
    neighbors[1:, :, :] |= film[:-1, :, :] > 0  # 上面
    neighbors[:-1, :, :] |= film[1:, :, :] > 0  # 下面
    neighbors[:, 1:, :] |= film[:, :-1, :] > 0  # 左边
    neighbors[:, :-1, :] |= film[:, 1:, :] > 0  # 右边
    neighbors[:, :, 1:] |= film[:, :, :-1] > 0  # 前面
    neighbors[:, :, :-1] |= film[:, :, 1:] > 0  # 后面

    # 获取满足条件的索引
    condition_depo = current_plane & neighbors

    # 更新表面稀疏张量
    surface_sparse_depo[condition_depo] = True

    return surface_sparse_depo

def scanZ_underSurface_bool(film):
    # 初始化一个全零的表面数组
    surface_sparse_depo = np.zeros_like(film, dtype=np.bool_)

    # depo 条件
    current_plane = film == 3

    # 创建邻居布尔索引数组
    neighbors = np.zeros_like(film, dtype=bool)

    # 获取周围邻居的布尔索引
    neighbors[1:, :, :] |= film[:-1, :, :] == 1  # 上面
    neighbors[:-1, :, :] |= film[1:, :, :] == 1  # 下面
    neighbors[:, 1:, :] |= film[:, :-1, :] == 1  # 左边
    neighbors[:, :-1, :] |= film[:, 1:, :] == 1  # 右边
    neighbors[:, :, 1:] |= film[:, :, :-1] == 1  # 前面
    neighbors[:, :, :-1] |= film[:, :, 1:] == 1  # 后面

    # 获取满足条件的索引
    condition_depo = current_plane & neighbors

    # 更新表面稀疏张量
    surface_sparse_depo[condition_depo] = True

    return surface_sparse_depo

def get_normal_from_index( film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point):
    x, y, z = point
    x += mirrorGap
    y += mirrorGap
    grid_cube = film_label_index_normal_mirror[x-3:x+4, y-3:y+4, z-3:z+4]

    plane_bool = grid_cube[:, :, :, 0] == 1
    positions = grid_cube[plane_bool][:, 1:4]

    xmn = np.mean(positions[:, 0])
    ymn = np.mean(positions[:, 1])
    zmn = np.mean(positions[:, 2])
    c = positions - np.stack([xmn, ymn, zmn])
    cov = np.dot(c.T, c)

    # SVD 分解协方差矩阵
    u, s, vh = np.linalg.svd(cov)

    # 最小特征值对应的特征向量
    normal = u[:, -1]  # 最小特征值的特征向量是最后一列

    x -= mirrorGap
    y -= mirrorGap
    film_label_index_normal[x, y, z, -3:] = normal
    return film_label_index_normal

def update_surface_mirror(surface_etching,surface_etching_mirror, mirrorGap, cellSizeX, cellSizeY):
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

def build_film_label_index_normal( sumfilm, mirrorGap):
    surface_film = scanZ_numpy_bool(sumfilm)
    vacuum_film = scanZ_vacuum_numpy_bool(sumfilm)

    film_label = np.zeros_like(sumfilm)

    solid_mask = sumfilm != 0
    film_label[solid_mask] = 3
    film_label[surface_film] = 1
    undersurface_film = scanZ_underSurface_bool(film_label)
    film_label[undersurface_film] = 2
    film_label[vacuum_film] = -1

    film_label_index_normal = np.zeros((film_label.shape[0], film_label.shape[1], film_label.shape[2], 7))

    for i in range(film_label_index_normal.shape[0]):
        for j in range(film_label_index_normal.shape[1]):
            for k in range(film_label_index_normal.shape[2]):
                film_label_index_normal[i, j, k, 0] = film_label[i, j, k]
                film_label_index_normal[i, j, k, 1] = i
                film_label_index_normal[i, j, k, 2] = j
                film_label_index_normal[i, j, k, 3] = k

    cellSizeX, cellSizeY, cellSizeZ = sumfilm.shape
    film_label_index_normal_mirror = np.zeros((cellSizeX+int(mirrorGap*2), cellSizeY+int(mirrorGap*2), cellSizeZ, 7))
    film_label_index_normal_mirror = update_surface_mirror(film_label_index_normal, film_label_index_normal_mirror, mirrorGap, cellSizeX, cellSizeY)

    surface_point = np.array(np.where(film_label_index_normal[:, :, :, 0] == 1)).T

    print(surface_point.shape)
    for i in range(surface_point.shape[0]):
        # print(surface_point[i])
        film_label_index_normal = get_normal_from_index(film_label_index_normal_mirror, film_label_index_normal, mirrorGap, surface_point[i])
    return film_label_index_normal


sumFilm = np.sum(film, axis=-1)
mirrorGap = 3
cellSizeX, cellSizeY, cellSizeZ = sumFilm.shape
film_label_index_normal = build_film_label_index_normal(sumFilm, mirrorGap)
film_label_index_normal_mirror = np.zeros((cellSizeX+int(mirrorGap*2), cellSizeY+int(mirrorGap*2), cellSizeZ, 7))
film_label_index_normal_mirror = update_surface_mirror(film_label_index_normal, film_label_index_normal_mirror, mirrorGap, cellSizeX, cellSizeY)


film_mirror = np.zeros((film.shape[0]+int(mirrorGap*2), film.shape[1]+int(mirrorGap*2), film.shape[2], film.shape[3]))
film_mirror= update_surface_mirror(film, film_mirror, mirrorGap, cellSizeX, cellSizeY)



cell = np.zeros((film_label_index_normal_mirror.shape[0], 
                 film_label_index_normal_mirror.shape[1], 
                 film_label_index_normal_mirror.shape[2]), dtype=Cell_dtype)

cell['id'] = film_label_index_normal_mirror[:,:,:,0]
cell['normal'] = film_label_index_normal_mirror[:,:,:,-3:]
cell['film'] = film_mirror
cell['index'] = film_label_index_normal_mirror[:,:,:,1:4]


simulation = SimProfile.Simulation(42, 26, 190, 670)

simulation.set_all_parameters(react_table_equation, react_type_table, react_prob_chemical, react_yield_p0, rn_coeffcients)
simulation.print_react_table_equation()

simulation.inputCell(cell['id'], cell['index'], cell['normal'], cell['film'])


def posGenerator_top_nolength(IN, cellSizeX, cellSizeY, cellSizeZ):
    emptyZ = 10
    position_matrix = np.array([np.random.rand(IN)*cellSizeX, \
                                np.random.rand(IN)*cellSizeY, \
                                np.random.uniform(0, emptyZ, IN) + cellSizeZ - emptyZ], dtype=np.double).T
    return position_matrix


N = int(1e6)
particle_list = [[N, 1, 'maxwell', 50]]
vel_matrix = particleGenerator.vel_generator(particle_list)

pos = posGenerator_top_nolength(N, 26, 190, 670)
# pos = posGenerator_top_nolength(N, 100, 100, 100)
# pos = posGenerator_top_nolength(N, 50, 100, 100)
pos = np.ascontiguousarray(pos, dtype=np.double)
vel = vel_matrix[:, :3].copy()
vel = np.ascontiguousarray(vel, dtype=np.double)
id  = vel_matrix[:, -1].astype(np.int32).copy()
E  = vel_matrix[:, -2].copy()

simulation.inputParticle(pos, vel, E, id)

simulation.runSimulation()