
import numpy as np






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

    #-----------------------------------------------------------------------------
    # # 1) 中心化
    # pts = positions - positions.mean(axis=0)  # shape = (N,3)

    # # 2) SVD -> pts = U · S · Vt
    # #    S 中是降序排列的奇异值 [s0, s1, s2]
    # _, S, Vt = np.linalg.svd(pts, full_matrices=False)

    # # 3) 奇异值变成协方差特征值： λi = (si^2) / N
    # lambdas = (S ** 2) / positions.shape[0]   # shape = (3,)
    # total = (lambdas.sum()) if lambdas.sum() > 0 else 0.0

    # # float(lambdas[-1] / lambdas.sum()) if lambdas.sum() > 0 else 0.0
    # # if total <= 0:
    # #     return 0.0

    # # 曲率大小
    # c_mag = lambdas[-1] / total  # 最小 λ / λ 总和

    # # 4) 最小主方向
    # v_min = Vt[-1, :]  # shape = (3,)

    # # 5) 与中心法线点积决定符号
    # # sign = np.sign(np.dot(center_normal, v_min))

    #-----------------------------------------------------------------------------

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
