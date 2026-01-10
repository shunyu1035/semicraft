from PIL import Image
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import postProcess as PostProcess
import particleGenerator as particleGenerator

import semicraft


def trainProfile(iter, preTrain_input, path):

    FILMSIZE = 4
    zoom_in = 1.4
    density = 20
    def inputFilm(jpg1):
        # image = cv2.imread("./sf_o2_20p_paper1.jpg")
        image = cv2.imread(jpg1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # zoom_in = 1.0
        # 调整图像大小，使它们具有相同的尺寸
        height, width = image.shape[:2]

        height = int(height/zoom_in)
        width = int(width/zoom_in)

        image = cv2.resize(image, (width, height))

        HardMasK = np.array(image)
        print("HardMasK.shape: ", HardMasK.shape)
        # FILMSIZE = 11 
        # Hard mask SF6_O2

        cellx = 60
        film = np.zeros((cellx, HardMasK.shape[1], HardMasK.shape[0], FILMSIZE), dtype=np.int32)
        # density = 20

        SiCut = 0
        for j in range(HardMasK.shape[0]):
        # for i in range(HardMasK.shape[1]):
            # print(j)
            # if HardMasK[j, 0] < 180 : # si
            #     SiCut = j

            for i in range(HardMasK.shape[1]):
                # if HardMasK[j, i] >= 100 and HardMasK[j, i] < 200: # hm
                if HardMasK[j, i] >= 100 and HardMasK[j, i] < 200 and i < HardMasK.shape[1]/2: # hm
                    # film[:, i, -j, 0] = density
                    SiCut = j
                    film[:, i, -j, -1] = density*5
                    film[:, -i, -j, -1] = density*5
                if HardMasK[j, i] > 180: # si
                    # film[:, i, -j, -1] = density*5
                    film[:, i, -j, 0] = density
            # print('SiCut: ', SiCut)

        # film[:, :, :HardMasK.shape[0] - SiCut, -1] = 0
        film[:, :, :HardMasK.shape[0] - SiCut+1, -1] = 0
        film[:, :, :HardMasK.shape[0] - SiCut+1, 0] = density

        # for x in range(film.shape[0]):
        #     for y in range(film.shape[1]):
        #         if film[x, y, HardMasK.shape[0] - SiCut, -1] != 0:
        #             print(f"x:{x}, y:{y}")
        #             film[x, y, HardMasK.shape[0] - SiCut, -2] = density  # undermask
        #             film[x, y, HardMasK.shape[0] - SiCut, -1] = 0  # undermask

        film[:, :, :5, 0] = density

        return film


    def getStop(jpg2):
        image2 = cv2.imread(jpg2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # zoom_in = 1.0
        # 调整图像大小，使它们具有相同的尺寸
        height, width = image2.shape[:2]

        height = int(height/zoom_in)
        width = int(width/zoom_in)

        image2 = cv2.resize(image2, (width, height))

        HardMasK2 = np.array(image2)
        print("HardMasK.shape: ", HardMasK2.shape)
        # FILMSIZE = 11 
        # Hard mask SF6_O2

        film2 = np.zeros((2, HardMasK2.shape[1], HardMasK2.shape[0], FILMSIZE), dtype=np.int32)
        # density = 20

        SiCut = 0
        for j in range(HardMasK2.shape[0]):
        # for i in range(HardMasK2.shape[1]):
            # print(j)
            if HardMasK2[j, 0] < 180 : # si
                SiCut = j

            for i in range(HardMasK2.shape[1]):
                if HardMasK2[j, i] >= 100 and HardMasK2[j, i] < 200: # hm
                    # film2[:, i, -j, 0] = density
                    film2[:, i, -j, -1] = density*5
                if HardMasK2[j, i] > 180: # si
                    # film2[:, i, -j, -1] = density*5
                    film2[:, i, -j, 0] = density
            # print('SiCut: ', SiCut)

        film2[:, :, :HardMasK2.shape[0] - SiCut, -1] = 0

        film2[:, :, :5, 0] = density

        stopPointY = int(film2.shape[1]/5)

        for thick in range(film2.shape[2]):
            if np.sum(film2[int(film2.shape[0]/2),stopPointY, thick, :]) == 0:
                stopPointZ = thick
                break

        for thick in range(film2.shape[2]):
            if np.sum(film2[int(film2.shape[0]/2), :, thick, :]) == 0:
                topLayerZ = thick
                break
            

        print('stopPointY: ', stopPointY)
        print('stopPointZ: ', stopPointZ)
        print('topLayerZ: ', topLayerZ)
        return stopPointY, stopPointZ, topLayerZ
    
    # 读取两张图像
    img1_path = "./sf_250410_1_1.jpg"
    img2_path = "./sf_250410_1_2.jpg"

    film = inputFilm(img1_path)
    stopPointY, stopPointZ, topLayerZ = getStop(img2_path)

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


    def get_normal_from_index( film_label_index_normal_mirror, film_label_index_normal, curvatures_field, mirrorGap, point):
        x, y, z = point
        x += mirrorGap
        y += mirrorGap
        grid_cube = film_label_index_normal_mirror[x-3:x+4, y-3:y+4, z-3:z+4]

        plane_bool = grid_cube[:, :, :, 0] == 1
        positions = grid_cube[plane_bool][:, 1:4]

        # get direction
        top = grid_cube[3, 3, 6, 0]
        bottom = grid_cube[3, 3, 0, 0]
        front = grid_cube[0, 3, 3, 0]
        back = grid_cube[6, 3, 3, 0]
        left = grid_cube[3, 0, 3, 0]
        right = grid_cube[3, 6, 3, 0]

        direction = np.array([0, 0, 1])

        if (top == 3 or top == 2) and (bottom == 0):
            direction = np.array([0, 0, -1])
        elif (front == 3 or front == 2) and (back == 0):
            direction = np.array([1, 0, 0])
        elif (back == 3 or back == 2) and (front == 0):
            direction = np.array([-1, 0, 0])
        elif (left == 3 or left == 2) and (right == 0):
            direction = np.array([0, 1, 0])
        elif (right == 3 or right == 2) and (left == 0):
            direction = np.array([0, -1, 0])

        # # 计算质心
        # centroid = positions.mean(axis=0)
        # # 构造协方差矩阵
        # cov = ((positions - centroid).T @ (positions - centroid)) / positions.shape[0]

        c = positions - positions.mean(axis=0)

        curvature_sign = point - positions.mean(axis=0)

        cov = np.dot(c.T, c)



        # SVD 分解协方差矩阵
        u, s, vh = np.linalg.svd(cov)

        # 最小特征值对应的特征向量
        normal = u[:, -1]  # 最小特征值的特征向量是最后一列

        dot = np.dot(normal, direction)

        normal = normal*np.sign(dot)

        # sign = np.sign(np.dot(normal, v_min))
        lambdas = s

        # principal_dir = vh[0]  # shape (3,)
        # dot_pt = np.dot(normal, principal_dir)

        # 曲率 = 最小 λ / (λ0+λ1+λ2)
        curvatures = lambdas[-1] / lambdas.sum()

        curvature_sign_dot = np.sign(np.dot(curvature_sign, normal))

        # curvatures_field[x, y, z] = curvatures
        x -= mirrorGap
        y -= mirrorGap
        # curvatures_field[x, y, z] = c_mag
        film_label_index_normal[x, y, z, -3:] = normal
        curvatures_field[x, y, z] = curvatures * curvature_sign_dot

        return film_label_index_normal, curvatures_field

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
        curvatures_field = np.zeros((film_label.shape[0], film_label.shape[1], film_label.shape[2]))
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
            film_label_index_normal, curvatures_field = get_normal_from_index(film_label_index_normal_mirror, film_label_index_normal, curvatures_field, mirrorGap, surface_point[i])
        return film_label_index_normal, curvatures_field

    sumFilm = np.sum(film, axis=-1)
    mirrorGap = 3
    cellSizeX, cellSizeY, cellSizeZ = sumFilm.shape
    film_label_index_normal, curvatures_field = build_film_label_index_normal(sumFilm, mirrorGap)

    curvatures_field[:3, :, :] = curvatures_field[3:6, :, :]
    curvatures_field[-3:, :, :] = curvatures_field[-6:-3, :, :]



    cellid = np.ascontiguousarray(film_label_index_normal[:,:,:,0].copy(), dtype=np.int32)
    cellnormal = np.ascontiguousarray(film_label_index_normal[:,:,:,-3:].copy(), dtype=np.double)
    cellfilm = np.ascontiguousarray(film.copy(), dtype=np.int32)
    cellindex = np.ascontiguousarray(film_label_index_normal[:,:,:,1:4].copy(), dtype=np.int32)



    react_table_equation =   np.array([[[-1,  1,  0,  0], #Si-SiFx
                                        [ 0, -1,  0,  0], #SiFx-vac
                                        [ 0,  0,  0,  0], #SiOxFy-F
                                        [ 0,  0,  0,  0]], #Mask-F

                                       [[-1,  0,  1,  0], #Si-SioxFy
                                        [ 0, -1,  1,  0], #SiFx-SioxFy
                                        [ 0,  0,  0,  0], #SioxFy-O
                                        [ 0,  0,  0,  0]], #Mask--O

                                       [[-1,  0,  0,  0], #Si-ion
                                        [ 0, -1,  0,  0], #SiFx-ion
                                        [ 0,  0, -1,  0], #SiOxFy-ion
                                        [ 0,  0,  0, -1]]], dtype=np.int32) #Mask-ion

    react_type_table =np.array([[1, 4, 1, 1],
                                [1, 1, 1, 1],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3]]) #sf6-o2-ion

    react_prob_chemical = np.array([[0.9, 0.5, 0.0, 0.0],
                                    [0.5, 0.5, 0.0, 0.0],
                                    [1.0, 1.0, 1.0, 1.0]])

    react_prob_chemical = np.ascontiguousarray(react_prob_chemical, dtype=np.double)

    # react_type_table = np.ones((4, 4), dtype=np.int32) # 1 for chemical transfer 
    # react_type_table[2, :] = 2 # 2 for physics
    # react_type_table[3, :] = 3 # 3 for redepo
    # react_type_table[0, 1] = 4 # 1 for chemical remove
    # # react_type_table[0, -1] = 4 # 1 for chemical remove
    # react_type_table[2, -1] = 2 # 2 for no reaction for mask in test

    print(' react_table_equation:')
    print(react_type_table)
    
    reflect_probability = np.ones((4, 4), dtype=np.double) # 1 for chemical transfer 
    reflect_coefficient = np.ones((4, 4), dtype=np.double) # 1 for chemical transfer 

    react_yield_p0 = np.array([0.00, 0.00, 0.30, 0.30])

    film_eth = np.array([15, 15, 15, 15], dtype=np.double)

    rn_coeffcients = np.array([[0.9423, 0.9434, 2.742, 3.026],
                                [0.9620, 0.9608, 2.542, 3.720],
                                [0.9458, 0.9445, 2.551, 3.735],
                                [1.046, 1.046, 2.686, 4.301]])

    # sputter_yield_coefficient = [0.3, 0.001, np.pi/4]
    sputter_yield_coefficient = np.array([[0.5536, 12.60258, 0.4457],
                                          [0.5536, 12.60258, 0.4457],
                                          [0.5536, 12.60258, 0.4457],
                                          [0.5536, 12.60258, 0.4457]])

    def posGenerator_top_nolength(IN, cellSizeX, cellSizeY, cellSizeZ, emptyZ):
        # emptyZ = 10
        position_matrix = np.array([np.random.rand(IN)*cellSizeX, \
                                    np.random.rand(IN)*cellSizeY, \
                                    np.random.uniform(0, emptyZ, IN) + cellSizeZ - emptyZ], dtype=np.double).T
        return position_matrix

    def overlap(n, img1_path, img2_path, overlap_range=np.array([[0, -1], [0, -1]])):
        # 读取两张图像
        # img2_path = "./sf_o2_20p_paper2.jpg"
        # img1_path = "./sf_o2_20p_paper1.jpg"

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # n = 20
        binary1 = np.load(f'./{path}/film_{n}_{iter}.npy')

        binary1 = np.sum(binary1, axis=-1)
        binary1 = binary1[int(binary1.shape[0]/2), :, :]

        new_binary1 = np.zeros((binary1.shape[1], binary1.shape[0]))
        print(new_binary1.shape)
        for i in range(binary1.shape[1]):
            new_binary1[new_binary1.shape[0] - i-1, :] = binary1[:, i]

        img1 = new_binary1


        # 调整图像大小，使它们具有相同的尺寸
        height, width = img1.shape[:2]
        img2_resized = cv2.resize(img2, (width, height))

        # 转换为灰度图
        # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        # _, binary1 = cv2.threshold(gray1, 80, 255, cv2.THRESH_BINARY)
        binary1 = new_binary1


        _, binary2 = cv2.threshold(gray2, 80, 255, cv2.THRESH_BINARY)

        # binary2 = cv2.bitwise_not(binary2).astype(np.uint8)
        # binary1 = binary1.astype(np.uint8)

        binary1 = (binary1 > 10).astype(np.uint8) * 255
        # binary2 = (binary2 > 128).astype(np.uint8) * 255

        # overlap_range = np.array([[80, 230], [20, 70]])
        x1 = overlap_range[0, 0]
        x2 = overlap_range[0, 1]
        y1 = overlap_range[1, 0]
        y2 = overlap_range[1, 1]
        binary1 = binary1[x1:x2, y1:y2]
        binary2 = binary2[x1:x2, y1:y2]
        # 计算重叠区域
        intersection = binary1 == binary2  # 计算交集

        overlap_ratio = np.sum(intersection) / (intersection.shape[0]*intersection.shape[1])

        return overlap_ratio
    
    # samples = 40
    np.random.seed(42)
    # preTrain_input = np.random.rand(40, 21)

    # overlap_output = np.zeros(preTrain_input.shape[0])
    # filmThickness = getBottom("./sf_250310_2.jpg")
    # for i in range(preTrain_input.shape[0]):
    i = 0
    E_decrease = np.zeros((3,4))

    # preTrain_input = np.load(f'./calibration_0922_bias20_paper/hyperopt/iteration_137.npy', allow_pickle=True).item()['params']
    while i < 1:
        # np.save(f"./train/preTrain_input_{i}_{iter}.npy", preTrain_input[i])
        seed = 125
        react_prob_chemical[0, 0] = 0.5
        react_prob_chemical[0, 1] = 0.5

        # react_prob_chemical[0, 0] = 0.0751224
        # react_prob_chemical[0, 1] = 0.9905068

        # react_prob_chemical[0, 0] = 0.73
        # react_prob_chemical[0, 1] = 0.61
        # react_prob_chemical[1, 0] = 0.06
        # react_prob_chemical[1, 1] = 0.09
        react_prob_chemical[1, 0] = 0.2
        react_prob_chemical[1, 1] = 0.2
        # react_prob_chemical[0, 3] = preTrain_input['react_prob_chemical_3'] #mask-vac

        # react_prob_chemical[1, 0] = preTrain_input['react_prob_chemical_10'] #Si-SioxFy
        # react_prob_chemical[1, 1] = preTrain_input['react_prob_chemical_11'] #SiFx-SioxFy
        reflect_coefficient[:, :] = 0.1
        reflect_coefficient[2, :] = preTrain_input['reflect_coefficient']   
        # reflect_coefficient = 0.10
        # chemical_angle_v1 = preTrain_input['chemical_angle_v1']
        # chemical_angle_v2 = preTrain_input['chemical_angle_v2']
        chemical_angle_v1 = np.pi/2 
        chemical_angle_v2 = np.pi/2 + 0.0005
        if chemical_angle_v1 > chemical_angle_v2:
            chemical_angle_v2 = chemical_angle_v1 + 0.001
        # chemical_angle_v2 = chemical_angle_v1 + 0.001
        sigma_ion = 0.05
        # Ar_ratio = preTrain_input['Ar_ratio']
        Ar_ratio = 0.23
        # o2_ratio = 0
        # simulation_step = preTrain_input['simulation_step']

        # 0.344679 1.89327 1.31794
        sputter_yield_coefficient[2, 0] = 0.054679
        sputter_yield_coefficient[2, 1] = 1.89327
        sputter_yield_coefficient[2, 2] = 0.31794

        sputter_yield_coefficient[3, 0] = preTrain_input['gamma_0']
        sputter_yield_coefficient[3, 1] = preTrain_input['f'] 
        sputter_yield_coefficient[3, 2] = preTrain_input['theta_max'] 
        # sputter_yield_coefficient[2, 1] = 1.79813
        # sputter_yield_coefficient[2, 2] = 1.3

        gamma0 = sputter_yield_coefficient[2, 0]
        f = sputter_yield_coefficient[2, 1]
        thetaMax = sputter_yield_coefficient[2, 2]
        # f = -np.log(gammaMax_2)/(np.log(np.cos(gammaMax_2*gamma0)) + 1 - np.cos(thetaMax))
        print('gamma0, f, thetaMax: ', gamma0, f, thetaMax)
        # print('f: ', f)
        # if f < 0:
        #     print("Simulation failed! f < 0")
        #     # np.save(f"./{path}/film_{i}_{iter}.npy", film)
        #     return 0
        # sputter_yield_coefficient[2, 0] = 0.6554548177413946
        # sputter_yield_coefficient[2, 1] = 2.06827
        # sputter_yield_coefficient[2, 2] = 1.2493461374795234


        # sputter_yield_coefficient[2, 2] = np.pi/6
        # if sputter_yield_coefficient[1] > sputter_yield_coefficient[0]:
        #     sputter_yield_coefficient[1] = sputter_yield_coefficient[0]
        print('sputter_yield_coefficient: ', sputter_yield_coefficient)
        print('chemical_angle_v1: ', chemical_angle_v1)
        print('chemical_angle_v2: ', chemical_angle_v2)
        # react_yield_p0_p = preTrain_input['react_yield_p0']
        # react_yield_p0[-1] = 0
        # react_yield_p0 *= react_yield_p0_p
        # react_yield_p0[0] = preTrain_input['react_yield_p0_0']
        # react_yield_p0[1] = preTrain_input['react_yield_p0_1']
        # react_yield_p0[2] = preTrain_input['react_yield_p0_2']
        react_yield_p0[2] = 1
        # react_yield_p0[3] = preTrain_input['react_yield_p0_3']
        # react_yield_p0[3] = 0.15
        react_yield_p0[3] = 0.5
        # react_yield_p0[2] = 0.6
        # react_yield_p0[3] = 0.20
        # react_yield_p0[9] = preTrain_input['react_yield_p0_9']
        # react_yield_p0[-1] = 0
        # E_decrease[:,:] = preTrain_input['E_decrease']
        # E_decrease[2,:] = 70
        # E_decrease[2,-1] = preTrain_input['E_decrease_ion_no_mask']

        E_decrease[:,:] = 21

        E_decrease[2,0] = 60
        E_decrease[2,1] = 60
        E_decrease[2,2] = 60

        E_decrease[2,-1] = 11

        # reflect_probability[0, :] = preTrain_input_PRE['reflect_probability_F']
        # reflect_probability[1, :] = preTrain_input_PRE['reflect_probability_O']
        # reflect_probability[2, :] = preTrain_input_PRE['reflect_probability_ion']
        # reflect_probability[2, 3] = preTrain_input_PRE['reflect_probability_mask_ion']

        reflect_probability[0, :] = -0.02968777
        reflect_probability[1, :] = 0.99090525
        # reflect_probability[2, :] = 0.26115128
        reflect_probability[2, :] = 0.07
        # reflect_probability[2, :] = 1
        reflect_probability[2, 3] = 1

        # reflect_coefficient[0, :] = 0.10
        # reflect_coefficient[1, :] = 0.10
        # reflect_coefficient[2, :] = 0.10
        # reflect_coefficient[2, 3] = 0.10
        # reflect_probability[2, :] = 0
        # reflect_coefficient[:, :] = 0.16
        # reflect_probability[2, 3] = 0
        

        # E_decrease = 60
        # react_yield_p0[10] = preTrain_input['react_yield_p0_10']
        print('react_prob_chemical: ', reflect_probability)
        print('E_decrease: ',E_decrease)
        print('react_prob_chemical: ', react_prob_chemical)
        print('react_yield_p0: ', react_yield_p0)
        # print('particle_ratio: ', particle_ratio)
        # Ar_ratio = 0
        # print('Ar_ratio', Ar_ratio)
        # print('sigma_ion', sigma_ion)
        print('reflect_coefficient: ',reflect_coefficient)

        simulation_step_rm = int(2e6)
        # simulation_step_rm = int(1e5)
        print(f"simulation_step[{iter}]: ", simulation_step_rm)

        N = int(9e6)

        o2_ratio = 0.5
        # particle_list = [[N, 0, 'cosn', 50, cosn]] # [int(N*Ar_ratio), 2, 'cosn', 50, cosn]
        # particle_list = [[int(N*(1-Ar_ratio)), 0, 'maxwell', 50, 300, 9], [int(N*Ar_ratio), 2, 'cosn', 50, cosn]]
        particle_list = [[int(N*(1-Ar_ratio)*(1-o2_ratio)), 0, 'maxwell', 50, 300, 9], [int(N*(1-Ar_ratio)*(o2_ratio)), 1, 'maxwell', 50, 300, 16], [int(N*Ar_ratio), 2, 'guass', 50, sigma_ion]]
        # particle_list = [[int(N), 0, 'maxwell', 50, 300, 9]]
        # particle_list = [[int(N), 0, 'maxwell_energy', 50, 300, 9]]

        vel_matrix = particleGenerator.vel_generator(particle_list)

        pos = posGenerator_top_nolength(vel_matrix.shape[0], cellSizeX, cellSizeY, cellSizeZ, (cellSizeZ - topLayerZ - 3))
        pos = np.ascontiguousarray(pos, dtype=np.double)

        vel = vel_matrix[:, :3].copy()
        vel = np.ascontiguousarray(vel, dtype=np.double)
        id  = vel_matrix[:, -1].astype(np.int32).copy()
        E  = vel_matrix[:, -2].copy()
        # **创建临时 pickle 文件**

        # max_particles = 256000
        # max_particles = 512000
        # max_particles = 1024000
        # max_particles = 2048000
        max_particles = 4096000
        simulation = semicraft.Simulation(seed, cellSizeX, cellSizeY, cellSizeZ, FILMSIZE, density, max_particles)

        simulation.set_all_parameters(react_table_equation, react_type_table, reflect_probability, reflect_coefficient,
                                    react_prob_chemical, react_yield_p0, 
                                    film_eth, rn_coeffcients, E_decrease)

        simulation.input_sputter_yield_coefficient(sputter_yield_coefficient)
        simulation.inputCell(cellid, curvatures_field, cellindex, cellnormal, cellfilm)
        simulation.inputParticle(pos, vel, E, id)


        depo_or_etch = -1 # depo for 1, etch for -1 
        ArgonID = 2
        diffusion = False
        diffusion_coeffient = 3.0
        diffusion_distant = 1
        redepo = False
        # reflect_probability = 0.1

        relaxTime = 100
        # **运行 Simulation**
        needrecompute = simulation.runSimulation(simulation_step_rm, ArgonID, depo_or_etch, redepo, diffusion, diffusion_coeffient, diffusion_distant, stopPointY, stopPointZ, chemical_angle_v1, chemical_angle_v2, relaxTime)
        # needrecompute = simulation.runSimulation(data["simulation_step_rm"], 2, data["reflect_coefficient"], data["E_decrease"], data["stopPointY"], data["stopPointZ"], data["chemical_angle_v1"], data["chemical_angle_v2"])
        print(f"needrecompute: {needrecompute}")

        if needrecompute == 0:
            print("Simulation finished")
            # typeID_array, film_array = simulation.cell_data_to_numpy()
            typeID_array, film_array, potential_array = simulation.cell_data_to_numpy()
            normal_array = simulation.normal_to_numpy()
            np.save(f"./{path}/film_{i}_{iter}.npy", film_array)
            # np.save(f"./{path}/typeID_{i}_{iter}.npy", typeID_array)
            # np.save(f"./{path}/normal_{i}_{iter}.npy", normal_array)
            print(f"save finished_{iter}_")

            # overlap_output = overlap(i, img1_path, img2_path)

            o1 = overlap(i, img1_path, img2_path, np.array([[90, 102], [40, 58]]))
            o2 = overlap(i, img1_path, img2_path, np.array([[75, 90], [0, 40]]))

            o3 = overlap(i, img1_path, img2_path, np.array([[90, 102], [-58, -40]]))
            o4 = overlap(i, img1_path, img2_path, np.array([[75, 90], [-40, -1]]))

            # overlap_show(n, "./sf_250410_1_1.jpg", "./sf_250410_1_2.jpg", path, np.array([[90, 102], [40, 58]]))
            # overlap_show(n, "./sf_250410_1_1.jpg", "./sf_250410_1_2.jpg", path, np.array([[75, 90], [0, 40]]))
            # o3 = overlap(i, img1_path, img2_path, np.array([[140, 210], [50, 60]]))

            overlap_output = ( o1 + o2  + o3 + o4) / 4
            i += 1

    return overlap_output


