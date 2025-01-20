import numpy as np
# from scipy.spatial import cKDTree
from pykdtree.kdtree import KDTree
import torch
from scipy import interpolate
from numba import jit, prange
import src.operations.mirror as mirror

def svd_torch(cov):
    cov_tensor = torch.tensor(cov)
    u, s, vh = torch.linalg.svd(cov_tensor, full_matrices=False)
    return u.numpy(), s.numpy(), vh.numpy()

@jit(nopython=True)
def svd_numba(cov):
    N, _, _ = cov.shape
    u_list = np.empty((N, 3, 3))  # 左奇异矩阵
    s_list = np.empty((N, 3))     # 奇异值
    vh_list = np.empty((N, 3, 3)) # 右奇异矩阵
    for i in range(N):
        u, s, vh = np.linalg.svd(cov[i])  # 对每个 (3, 3) 矩阵计算 SVD
        u_list[i] = u
        s_list[i] = s
        vh_list[i] = vh
    
    # 计算最小奇异值的对应向量
    normal_all = np.empty((N, 3))  # 用于存储最终的 normal_all
    for i in range(N):
        minevindex = np.argmin(s_list[i])  # 找到最小奇异值的索引
        normal_all[i] = u_list[i, :, minevindex]  # 提取对应的向量
    
    return normal_all

@jit(nopython=True)
def eigen_min_numba(u, s, N):
    normal_all = np.empty((N, 3))  # 用于存储最终的 normal_all
    for i in range(N):
        minevindex = np.argmin(s[i])  # 找到最小奇异值的索引
        normal_all[i] = u[i, :, minevindex]  # 提取对应的向量
    
    return normal_all

@jit(nopython=True)
def min_eigenvector(cov):
    """
    获取每个 3x3 矩阵的最小特征值对应的特征向量
    :param cov: 输入矩阵 (N, 3, 3)
    :return: 最小特征值对应的特征向量数组 (N, 3)
    """
    N = cov.shape[0]
    vectors = np.empty((N, 3))
    for i in range(N):
        # 计算特征值和特征向量
        eigvals, eigvecs = np.linalg.eigh(cov[i])
        # 找到最小特征值的索引
        min_idx = np.argmin(eigvals)
        # 对应的特征向量
        vectors[i] = eigvecs[:, min_idx]
    return vectors



def find_adjacent_etching(film, indice):
    cell_ijk = np.array(np.nonzero(indice)).T
    cell_adjacent = np.zeros((6,3))
    cell_adjacent[0, 1] = cell_ijk

def check_valid(point, shape):
    mask_valid = (
        (point[:, :,0] >= 0) & (point[:, :,0] < shape[0]) &
        (point[:, :,1] >= 0) & (point[:, :,1] < shape[1]) &
        (point[:, :,2] >= 0) & (point[:, :,2] < shape[2])
    )
    point_nn_valid = point[mask_valid]
    return point_nn_valid


@jit(nopython=True)
def get_normal_from_index_numba(film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point):
    x, y, z = point
    x += mirrorGap
    y += mirrorGap

    # 提取 7x7x7 的局部区域
    grid_cube = film_label_index_normal_mirror[x-3:x+4, y-3:y+4, z-3:z+4]

    # 使用布尔掩码提取满足条件的索引
    plane_bool = grid_cube[:, :, :, 0] == 1

    # 预分配 positions 数组
    max_positions = np.sum(plane_bool)  # 最大可能点数
    positions = np.zeros((max_positions, 3))  # 假设每个点有 3 个坐标

    idx = 0  # 索引计数器
    for i in range(grid_cube.shape[0]):
        for j in range(grid_cube.shape[1]):
            for k in range(grid_cube.shape[2]):
                if plane_bool[i, j, k]:
                    positions[idx, :] = grid_cube[i, j, k, 1:4]
                    idx += 1

    # 截取有效部分
    positions = positions[:idx, :]

    if positions.shape[0] == 0:
        return film_label_index_normal  # 如果没有点，直接返回

    # 计算法向量
    xmn = np.mean(positions[:, 0])
    ymn = np.mean(positions[:, 1])
    zmn = np.mean(positions[:, 2])
    c = positions - np.array([xmn, ymn, zmn])
    cov = np.dot(c.T, c)

    # SVD 分解协方差矩阵
    u, s, vh = np.linalg.svd(cov)

    # 最小特征值对应的特征向量
    normal = u[:, -1]  # 最小特征值的特征向量是最后一列

    # 更新法向量到 film_label_index_normal
    x -= mirrorGap
    y -= mirrorGap
    film_label_index_normal[x, y, z, -3:] = normal

    return film_label_index_normal

@jit(nopython=True)
def unique_rows(data):
    """手动实现二维数组按行去重"""
    unique_data = []
    seen = set()
    for i in range(data.shape[0]):
        row = data[i]
        # 将数组的每行转为 `tuple` 的替代实现：用字符串表示
        row_key = ",".join(map(str, row))
        if row_key not in seen:
            seen.add(row_key)
            unique_data.append(row)
    return np.array(unique_data, dtype=data.dtype)

@jit(nopython=True)
def update_normal_in_matrix_numba(film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point_to_change):
    max_points = 100000  # 预分配的最大点数，可以根据需要调整
    point_nn_all = np.empty((max_points, 3), dtype=np.int64)  # 预分配数组
    current_index = 0  # 当前写入位置

    for i in range(point_to_change.shape[0]):
        x, y, z = point_to_change[i]
        x += mirrorGap
        y += mirrorGap

        # 提取邻域的 7x7x7 块
        grid_cube = film_label_index_normal_mirror[x - 3:x + 4, y - 3:y + 4, z - 3:z + 4]

        # 使用 np.where 提取满足条件的位置
        plane_bool = grid_cube[:, :, :, 0] == 1
        indices = np.where(plane_bool)  # 获取满足条件的索引 (3D)

        # 将索引转换为点并调整偏移
        for j in range(indices[0].shape[0]):
            new_point = np.array(
                [indices[0][j] - 3 + x - mirrorGap,
                 indices[1][j] - 3 + y - mirrorGap,
                 indices[2][j] - 3 + z - mirrorGap]
            )
            point_nn_all[current_index] = new_point
            current_index += 1

            # 防止数组溢出
            if current_index >= max_points:
                raise ValueError("Too many points, increase max_points.")

    # 去重并裁剪到实际使用的大小
    point_nn_all = point_nn_all[:current_index]

    # 更新法向量
    for i in range(point_nn_all.shape[0]):
        film_label_index_normal = get_normal_from_index_numba(
            film_label_index_normal_mirror,
            film_label_index_normal_mirror[mirrorGap:-mirrorGap, mirrorGap:-mirrorGap, :],
            mirrorGap,
            point_nn_all[i],
        )

    return film_label_index_normal

# def update_normal_in_matrix_numba(film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point_to_change):
#     point_nn_all = np.zeros((1,3))
#     # film_label_index_normal[point_to_change[:,0],point_to_change[:,1],point_to_change[:,2], 4:] = 0
#     for i in range(point_to_change.shape[0]):
#         x, y, z = point_to_change[i]
#         x += mirrorGap
#         y += mirrorGap
#         grid_cube = film_label_index_normal_mirror[x-3:x+4, y-3:y+4, z-3:z+4]
#         plane_bool = grid_cube[:, :, :, 0] == 1
#         point_nn = grid_cube[plane_bool][:, 1:4]
#         point_nn_all = np.vstack((point_nn_all, point_nn))
#     point_nn_all = np.unique(point_nn_all[1:], axis=0).astype(np.int64)
#     for i in range(point_nn_all.shape[0]):
#         film_label_index_normal = get_normal_from_index_numba(film_label_index_normal_mirror, film_label_index_normal_mirror[mirrorGap:-mirrorGap, mirrorGap:-mirrorGap, :], mirrorGap, point_nn_all[i])
#     return film_label_index_normal


class surface_normal:
    def __init__(self, center_with_direction, range3D, InOrOut, celllength, yield_hist = None):
        # center xyz inOrout
        self.center_with_direction = center_with_direction
        # boundary x1x2 y1y2 z1z2
        self.range3D = range3D 
        self.celllength = celllength
        # In for +1 out for -1
        self.InOrOut = InOrOut 
        if yield_hist.all() == None:
            self.yield_hist = np.array([[1.0, 1.05,  1.2,  1.4,  1.5, 1.07, 0.65, 0.28, 0.08,  0], \
                                        [0, np.pi/18, np.pi/9, np.pi/6, 2*np.pi/9, 5*np.pi/18, np.pi/3, 7*np.pi/18, 4*np.pi/9, np.pi/2]])
        else:
            self.yield_hist = yield_hist
        self.yield_func = interpolate.interp1d(self.yield_hist[1], self.yield_hist[0], kind='quadratic')


    
    def scanZ(self, film): # fast scanZ
        film = torch.Tensor(film)
        xshape, yshape, zshape = film.shape
        
        # 初始化一个全零的表面稀疏张量
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        
        # 获取当前平面与前后平面的布尔索引
        current_plane = film != 0

        # 获取周围邻居的布尔索引
        neighbors = torch.zeros_like(film, dtype=torch.bool)
        
        neighbors[1:, :, :] |= film[:-1, :, :] == 0  # 上面
        neighbors[:-1, :, :] |= film[1:, :, :] == 0  # 下面
        neighbors[:, 1:, :] |= film[:, :-1, :] == 0  # 左边
        neighbors[:, :-1, :] |= film[:, 1:, :] == 0  # 右边
        neighbors[:, :, 1:] |= film[:, :, :-1] == 0  # 前面
        neighbors[:, :, :-1] |= film[:, :, 1:] == 0  # 后面
        
        # 获取满足条件的索引
        condition = current_plane & neighbors
        
        # 更新表面稀疏张量
        surface_sparse[condition] = 1
        
        return surface_sparse.to_sparse()
        
    def scanZ_vaccum(self, film): # fast scanZ
        film = torch.Tensor(film)
        xshape, yshape, zshape = film.shape
        
        # 初始化一个全零的表面稀疏张量
        surface_sparse = torch.zeros((xshape, yshape, zshape))
        
        # 获取当前平面与前后平面的布尔索引
        current_plane = film == 0

        # 获取周围邻居的布尔索引
        neighbors = torch.zeros_like(film, dtype=torch.bool)
        
        neighbors[1:, :, :] |= film[:-1, :, :] > 0  # 上面
        neighbors[:-1, :, :] |= film[1:, :, :] > 0  # 下面
        neighbors[:, 1:, :] |= film[:, :-1, :] > 0  # 左边
        neighbors[:, :-1, :] |= film[:, 1:, :] > 0  # 右边
        neighbors[:, :, 1:] |= film[:, :, :-1] > 0  # 前面
        neighbors[:, :, :-1] |= film[:, :, 1:] > 0  # 后面
        
        # 获取满足条件的索引
        condition = current_plane & neighbors
        
        # 更新表面稀疏张量
        surface_sparse[condition] = 1
        
        return surface_sparse.to_sparse()

    def scanZ_numpy(self, film):
        # 初始化一个全零的表面数组
        surface_sparse_depo = np.zeros_like(film)

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
        surface_sparse_depo[condition_depo] = 1

        return surface_sparse_depo

    def scanZ_vacuum_numpy(self, film):
        # 初始化一个全零的表面数组
        surface_sparse_depo = np.zeros_like(film)

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
        surface_sparse_depo[condition_depo] = 1

        return surface_sparse_depo
    
    def scanZ_underSurface_bool(self, film):
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
    
    def get_normal_from_grid(self, film, normal_matrix, mirrorGap, point):
        # point += mirrorGap
        x, y, z = point
        x += mirrorGap
        y += mirrorGap
        grid_cube = film[x-3:x+4, y-3:y+4, z-3:z+4]

        positions = np.array(np.where(grid_cube == 1)).T

        xmn = np.mean(positions[:, 0])
        ymn = np.mean(positions[:, 1])
        zmn = np.mean(positions[:, 2])
        # print(f'xyzMin:{np.array([xmn, ymn, zmn])}')
        c = positions - np.stack([xmn, ymn, zmn])
        cov = np.dot(c.T, c)

        # SVD 分解协方差矩阵
        u, s, vh = np.linalg.svd(cov)

        # 最小特征值对应的特征向量
        normal = u[:, -1]  # 最小特征值的特征向量是最后一列
        x -= mirrorGap
        y -= mirrorGap
        normal_matrix[x, y, z] = normal
        return normal_matrix

    def get_normal_from_index(self, film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point):
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

    def get_normal_from_index_padded(self, film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point):
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

    # film_label_index_normal[label, x. y, z, nx, ny, nz]
    def build_film_label_index_normal(self, sumfilm, mirrorGap):
        surface_film = self.scanZ_numpy_bool(sumfilm)
        vacuum_film = self.scanZ_vacuum_numpy_bool(sumfilm)

        film_label = np.zeros_like(sumfilm)

        solid_mask = sumfilm != 0
        film_label[solid_mask] = 3
        film_label[surface_film] = 1
        undersurface_film = self.scanZ_underSurface_bool(film_label)
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
        film_label_index_normal_mirror = mirror.update_surface_mirror(film_label_index_normal, film_label_index_normal_mirror, mirrorGap, cellSizeX, cellSizeY)

        surface_point = np.array(np.where(film_label_index_normal[:, :, :, 0] == 1)).T

        print(surface_point.shape)
        for i in range(surface_point.shape[0]):
            # print(surface_point[i])
            film_label_index_normal = self.get_normal_from_index(film_label_index_normal_mirror, film_label_index_normal, mirrorGap, surface_point[i])
        return film_label_index_normal

    def build_normal_matrix(self, film, mirrorGap):
        surface_film = self.scanZ_numpy(film)
        vacuum_film = self.scanZ_vacuum_numpy(film)
        cellSizeX, cellSizeY, cellSizeZ = surface_film.shape

        surface_mirror = np.zeros((cellSizeX+int(mirrorGap*2), cellSizeY+int(mirrorGap*2), cellSizeZ))
        film_mirror = mirror.update_surface_mirror(surface_film, surface_mirror, mirrorGap, cellSizeX, cellSizeY)

        # normal_array shape(x,y,z,4) [solid_surface_vaccum(2, 1, -1, 0), nx, ny, ny] to film shape(x,y,z)
        normal_matrix = np.zeros((cellSizeX, cellSizeY, cellSizeZ, 3))
        surface_point = np.array(np.where(surface_film == 1)).T

        for i in range(surface_point.shape[0]):
            normal_matrix = self.get_normal_from_grid(film_mirror, normal_matrix, mirrorGap, surface_point[i])
        return normal_matrix, vacuum_film

    def update_normal_matrix(self, film_mirror, normal_matrix, mirrorGap, point_to_change):
        point_nn_all = np.zeros((1,3))
        normal_matrix[point_to_change[:,0],point_to_change[:,1],point_to_change[:,2], :] = 0
        for i in range(point_to_change.shape[0]):
            x, y, z = point_to_change[i]
            x += mirrorGap
            y += mirrorGap
            grid_cube = film_mirror[x-3:x+4, y-3:y+4, z-3:z+4]
            point_nn = np.array(np.where(grid_cube == 1)).T
            point_nn[:] -= 3
            point_nn += point_to_change[i]
            point_nn_all = np.vstack((point_nn_all, point_nn))
        point_nn_all = np.unique(point_nn_all[1:], axis=0).astype(np.int64)
        # print(point_nn_all)
        for i in range(point_nn_all.shape[0]):
            normal_matrix = self.get_normal_from_grid(film_mirror, normal_matrix, mirrorGap, point_nn_all[i])
        return normal_matrix

    def update_film_label_index_normal_etch(self, film_label_index_normal_pad, mirrorGap, point):
        grid_cross = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1,0],
                            [0,0,  1],
                            [0, 0,-1]])

        point[:, 0] += mirrorGap
        point[:, 1] += mirrorGap
        film_label_index_normal_pad[point[:, 0], point[:, 1], point[:, 2], 0] = -1 #etch

        # 向量化处理
        # 1. 计算所有点的邻居
        point_nn = (point[:, np.newaxis, :] + grid_cross).reshape(-1, 3)
        # print(f'point_nn: {point_nn}')
        # print(point_nn)
        # point_nn = check_valid(point_nn, shape)
        # 2. 筛选邻居点对应的 film_label 值为 2 的点
        # print(f'before_film_point_nn: {film_label_index_normal_pad[point_nn[:, 0], point_nn[:,1], point_nn[:,2], 0]}')

        mask = film_label_index_normal_pad[point_nn[:, 0], point_nn[:,1], point_nn[:,2], 0] == 2
        # print(f'film_point_nn: {film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0]}')
        # 3. 更新这些点对应的值为 1
        film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0] = 1
        # print(f'film_point_nn: {film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0]}')
        # 4. 筛选邻居点对应的 film_label 值为 -1 的点
        mask_vacuum = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == -1

        point_vacuum = point_nn[mask_vacuum]
        point_vacuum_nn = (point_vacuum[:, np.newaxis, :] + grid_cross)
        # print(f'point_vacuum_nn: {point_vacuum_nn}')
        # point_vacuum_nn = check_valid(point_vacuum_nn, shape)

        # 获取邻居的值
        neighbor_values = film_label_index_normal_pad[
            point_vacuum_nn[:,:, 0],
            point_vacuum_nn[:,:, 1],
            point_vacuum_nn[:,:, 2],
            0]
        # print(f'neighbor_values: {neighbor_values}')
        # 找出所有邻居都不等于 1 的点
        no_neighbor_equal_1 = ~np.any(neighbor_values == 1, axis=1)
        # print(point_vacuum[no_neighbor_equal_1])
        # 更新满足条件的点为 0
        film_label_index_normal_pad[point_vacuum[no_neighbor_equal_1, 0],
                point_vacuum[no_neighbor_equal_1, 1],
                point_vacuum[no_neighbor_equal_1, 2], 0] = 0
        
        # 5. 筛选邻居点对应的 film_label 值为 2 的点
        point_undersurface = (point_nn[mask, np.newaxis, :] + grid_cross).reshape(-1, 3)

        # 2. 筛选邻居点对应的 film_label 值为 3 的点
        mask_undersurface = film_label_index_normal_pad[point_undersurface[:, 0], point_undersurface[:, 1], point_undersurface[:, 2], 0] == 3
        # 3. 更新这些点对应的值为 2
        film_label_index_normal_pad[point_undersurface[mask_undersurface, 0], point_undersurface[mask_undersurface, 1], point_undersurface[mask_undersurface, 2], 0] = 2

        return film_label_index_normal_pad[mirrorGap:-mirrorGap,mirrorGap:-mirrorGap,:], film_label_index_normal_pad

    def update_film_label_index_normal_depo(self, film_label_index_normal_pad, mirrorGap, point):
        grid_cross = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1,0],
                            [0,0,  1],
                            [0, 0,-1]])
        
        point[:, 0] += mirrorGap
        point[:, 1] += mirrorGap
        film_label_index_normal_pad[point[:, 0], point[:, 1], point[:, 2], 0] = 1 #depo

        # 向量化处理
        # 1. 计算所有点的邻居
        point_nn = (point[:, np.newaxis, :] + grid_cross).reshape(-1, 3)
        # point_nn = check_valid(point_nn, shape)

        # 2. 筛选邻居点对应的 film_label 值为 0 的点
        mask = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == 0

        # 3. 更新这些点对应的值为 -1
        film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0] = -1

        # 4. 筛选邻居点对应的 film_label 值为 1 的点
        mask_vacuum = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == 1
        point_vacuum = point_nn[mask_vacuum]
        point_vacuum_nn = (point_vacuum[:, np.newaxis, :] + grid_cross)
        # point_vacuum_nn = check_valid(point_vacuum_nn, shape)

        # 获取邻居的值
        neighbor_values = film_label_index_normal_pad[
            point_vacuum_nn[:,:, 0],
            point_vacuum_nn[:,:, 1],
            point_vacuum_nn[:,:, 2],
            0]

        # 找出所有邻居都不等于 -1 的点
        no_neighbor_equal_1 = ~np.any(neighbor_values == -1, axis=1)

        # 更新满足条件的点为 2
        film_label_index_normal_pad[point_vacuum[no_neighbor_equal_1, 0],
                point_vacuum[no_neighbor_equal_1, 1],
                point_vacuum[no_neighbor_equal_1, 2], 0] = 2
        
        return film_label_index_normal_pad[mirrorGap:-mirrorGap,mirrorGap:-mirrorGap,:], film_label_index_normal_pad

    def update_normal_in_matrix(self, film_label_index_normal_mirror, film_label_index_normal, mirrorGap, point_to_change):
        point_nn_all = np.zeros((1,3))
        # film_label_index_normal[point_to_change[:,0],point_to_change[:,1],point_to_change[:,2], 4:] = 0
        for i in range(point_to_change.shape[0]):
            x, y, z = point_to_change[i]
            x += mirrorGap
            y += mirrorGap
            grid_cube = film_label_index_normal_mirror[x-3:x+4, y-3:y+4, z-3:z+4]
            plane_bool = grid_cube[:, :, :, 0] == 1
            point_nn = grid_cube[plane_bool][:, 1:4]
            point_nn_all = np.vstack((point_nn_all, point_nn))
        point_nn_all = np.unique(point_nn_all[1:], axis=0).astype(np.int64)
        for i in range(point_nn_all.shape[0]):
            film_label_index_normal = self.get_normal_from_index(film_label_index_normal_mirror, film_label_index_normal_mirror[mirrorGap:-mirrorGap, mirrorGap:-mirrorGap, :], mirrorGap, point_nn_all[i])
            # film_label_index_normal = get_normal_from_index_numba(film_label_index_normal_mirror, film_label_index_normal_mirror[mirrorGap:-mirrorGap, mirrorGap:-mirrorGap, :], mirrorGap, point_nn_all[i])
        return film_label_index_normal


    # ------------------------------------------------------------------------
    #   12/24 pointcloud也要动态更新在update里面
    # -----------------------------------------

    # plane_pointcloud (x,y,z) int64
    def build_pointcloud_from_matrix(self, film_label_index_normal):
        labels = film_label_index_normal[:, :, :, 0]
        plane_bool = labels == 1
        vacuum_bool = labels == -1
        plane_indices = np.argwhere(plane_bool)
        vacuum_indices = np.argwhere(vacuum_bool)
        plane_pointcloud_numpy = film_label_index_normal[plane_indices[:, 0], plane_indices[:, 1], plane_indices[:, 2], 1:4].astype(np.int64)
        vacuum_pointcloud_numpy = film_label_index_normal[vacuum_indices[:, 0], vacuum_indices[:, 1], vacuum_indices[:, 2], 1:4].astype(np.int64)

        plane_pointcloud_hash = {}
        vacuum_pointcloud_hash = {}

        for i in range(plane_pointcloud_numpy.shape[0]):
            plane_pointcloud_hash[(plane_pointcloud_numpy[i, 0], plane_pointcloud_numpy[i, 1], plane_pointcloud_numpy[i, 2])] = plane_pointcloud_numpy[i]
        for j in range(vacuum_pointcloud_numpy.shape[0]):
            vacuum_pointcloud_hash[(vacuum_pointcloud_numpy[j, 0], vacuum_pointcloud_numpy[j, 1], vacuum_pointcloud_numpy[j, 2])] = vacuum_pointcloud_numpy[j]

        return plane_pointcloud_hash, vacuum_pointcloud_hash


    def update_film_label_index_normal_etch_hash(self, film_label_index_normal_pad, mirrorGap, point, plane_pointcloud_hash, vacuum_pointcloud_hash):
        grid_cross = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1,0],
                            [0,0,  1],
                            [0, 0,-1]])

        point[:, 0] += mirrorGap
        point[:, 1] += mirrorGap
        film_label_index_normal_pad[point[:, 0], point[:, 1], point[:, 2], 0] = -1 #etch
        for i in range(point.shape[0]):
            vacuum_pointcloud_hash[(point[i, 0], point[i, 1], point[i, 2])] = point[i]
        # 向量化处理
        # 1. 计算所有点的邻居
        point_nn = (point[:, np.newaxis, :] + grid_cross).reshape(-1, 3)
        # print(f'point_nn: {point_nn}')
        # print(point_nn)
        # point_nn = check_valid(point_nn, shape)
        # 2. 筛选邻居点对应的 film_label 值为 2 的点
        # print(f'before_film_point_nn: {film_label_index_normal_pad[point_nn[:, 0], point_nn[:,1], point_nn[:,2], 0]}')

        mask = film_label_index_normal_pad[point_nn[:, 0], point_nn[:,1], point_nn[:,2], 0] == 2
        # print(f'film_point_nn: {film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0]}')
        # 3. 更新这些点对应的值为 1
        film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0] = 1
        for i in range(point_nn[mask].shape[0]):
            plane_pointcloud_hash[(point_nn[mask][i, 0], point_nn[mask][i, 1], point_nn[mask][i, 2])] = point_nn[mask][i]
        # print(f'film_point_nn: {film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0]}')
        # 4. 筛选邻居点对应的 film_label 值为 -1 的点
        mask_vacuum = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == -1

        point_vacuum = point_nn[mask_vacuum]
        point_vacuum_nn = (point_vacuum[:, np.newaxis, :] + grid_cross)
        # print(f'point_vacuum_nn: {point_vacuum_nn}')
        # point_vacuum_nn = check_valid(point_vacuum_nn, shape)

        # 获取邻居的值
        neighbor_values = film_label_index_normal_pad[
            point_vacuum_nn[:,:, 0],
            point_vacuum_nn[:,:, 1],
            point_vacuum_nn[:,:, 2],
            0]
        # print(f'neighbor_values: {neighbor_values}')
        # 找出所有邻居都不等于 1 的点
        no_neighbor_equal_1 = ~np.any(neighbor_values == 1, axis=1)
        # print(point_vacuum[no_neighbor_equal_1])
        # 更新满足条件的点为 0
        film_label_index_normal_pad[point_vacuum[no_neighbor_equal_1, 0],
                point_vacuum[no_neighbor_equal_1, 1],
                point_vacuum[no_neighbor_equal_1, 2], 0] = 0
        
        for i in range(point_vacuum[no_neighbor_equal_1].shape[0]):
            vacuum_pointcloud_hash.pop((point_vacuum[no_neighbor_equal_1][i, 0], point_vacuum[no_neighbor_equal_1][i, 1], point_vacuum[no_neighbor_equal_1][i, 2]), None)

        return film_label_index_normal_pad[mirrorGap:-mirrorGap,mirrorGap:-mirrorGap,:], film_label_index_normal_pad, plane_pointcloud_hash, vacuum_pointcloud_hash

    def update_film_label_index_normal_depo_hash(self, film_label_index_normal_pad, mirrorGap, point, plane_pointcloud_hash, vacuum_pointcloud_hash):
        grid_cross = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1,0],
                            [0,0,  1],
                            [0, 0,-1]])
        
        point[:, 0] += mirrorGap
        point[:, 1] += mirrorGap
        film_label_index_normal_pad[point[:, 0], point[:, 1], point[:, 2], 0] = 1 #depo
        for i in range(point.shape[0]):
            plane_pointcloud_hash[(point[i, 0], point[i, 1], point[i, 2])] = point[i]
        # 向量化处理
        # 1. 计算所有点的邻居
        point_nn = (point[:, np.newaxis, :] + grid_cross).reshape(-1, 3)
        # point_nn = check_valid(point_nn, shape)

        # 2. 筛选邻居点对应的 film_label 值为 0 的点
        mask = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == 0

        # 3. 更新这些点对应的值为 -1
        film_label_index_normal_pad[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 0] = -1
        for i in range(point_nn[mask].shape[0]):
            vacuum_pointcloud_hash[(point_nn[mask][i, 0], point_nn[mask][i, 1], point_nn[mask][i, 2])] = point_nn[mask][i]
        # 4. 筛选邻居点对应的 film_label 值为 1 的点
        mask_vacuum = film_label_index_normal_pad[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == 1
        point_vacuum = point_nn[mask_vacuum]
        point_vacuum_nn = (point_vacuum[:, np.newaxis, :] + grid_cross)
        # point_vacuum_nn = check_valid(point_vacuum_nn, shape)

        # 获取邻居的值
        neighbor_values = film_label_index_normal_pad[
            point_vacuum_nn[:,:, 0],
            point_vacuum_nn[:,:, 1],
            point_vacuum_nn[:,:, 2],
            0]

        # 找出所有邻居都不等于 -1 的点
        no_neighbor_equal_1 = ~np.any(neighbor_values == -1, axis=1)

        # 更新满足条件的点为 2
        film_label_index_normal_pad[point_vacuum[no_neighbor_equal_1, 0],
                point_vacuum[no_neighbor_equal_1, 1],
                point_vacuum[no_neighbor_equal_1, 2], 0] = 2

        for i in range(point_vacuum[no_neighbor_equal_1].shape[0]):
            plane_pointcloud_hash.pop((point_vacuum[no_neighbor_equal_1][i, 0], point_vacuum[no_neighbor_equal_1][i, 1], point_vacuum[no_neighbor_equal_1][i, 2]), None)

    
        return film_label_index_normal_pad[mirrorGap:-mirrorGap,mirrorGap:-mirrorGap,:], film_label_index_normal_pad, plane_pointcloud_hash, vacuum_pointcloud_hash

    def get_inject_normal_kdtree_hash(self, plane, plane_vaccum, pos):
        plane_point = plane[:, 1:4]
        normal = plane[:, -3:]
        plane_tree = KDTree(plane_point*self.celllength)
        dd, ii = plane_tree.query(pos, k=1)
        plane_point_int = np.array(plane_point[ii]).astype(int)
        
        plane_vaccum_tree = KDTree(plane_vaccum[:, 1:4]*self.celllength)
        dd_v, ii_v = plane_vaccum_tree.query(pos, k=1)
        plane_point_vaccum_int = np.array(plane_vaccum[ii_v, 1:4]).astype(int)

        return plane_point_int, normal[ii], plane_point_vaccum_int

    # -----------------------------------------
    #   12/24 pointcloud也要动态更新在update里面
    # ------------------------------------------------------------------------


    def update_film_label_etch(self, film_label, point):
        grid_cross = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1,0],
                            [0,0,  1],
                            [0, 0,-1]])

        film_label[point[:, 0], point[:, 1], point[:, 2]] = -1 #etch

        # 向量化处理
        # 1. 计算所有点的邻居
        point_nn = (point[:, np.newaxis, :] + grid_cross).reshape(-1, 3)

        # 2. 筛选邻居点对应的 film_label 值为 2 的点
        mask = film_label[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2]] == 2

        # 3. 更新这些点对应的值为 1
        film_label[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2]] = 1

        # 4. 筛选邻居点对应的 film_label 值为 -1 的点
        mask_vacuum = film_label[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2]] == -1

        point_vacuum = point_nn[mask_vacuum]
        point_vacuum_nn = (point_vacuum[:, np.newaxis, :] + grid_cross)

        # 获取邻居的值
        neighbor_values = film_label[
            point_vacuum_nn[:, :, 0],
            point_vacuum_nn[:, :, 1],
            point_vacuum_nn[:, :, 2]
        ]

        # 找出所有邻居都不等于 1 的点
        no_neighbor_equal_1 = ~np.any(neighbor_values == 1, axis=1)

        # 更新满足条件的点为 0
        film_label[point_vacuum[no_neighbor_equal_1, 0],
                point_vacuum[no_neighbor_equal_1, 1],
                point_vacuum[no_neighbor_equal_1, 2]] = 0
        
        return film_label

    def scanZ_numpy_bool(self, film):
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

    def scanZ_vacuum_numpy_bool(self, film):
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

    def initial_film_label(self, sumfilm):
        surface_film = self.scanZ_numpy_bool(sumfilm)
        vacuum_film = self.scanZ_vacuum_numpy_bool(sumfilm)

        film_label = np.zeros_like(sumfilm, dtype=np.int64)

        solid_mask = sumfilm != 0
        film_label[solid_mask] = 2
        film_label[surface_film] = 1
        film_label[vacuum_film] = -1
        return film_label
    
    def get_inject_normal_kdtree(self, plane, plane_vaccum, pos):
        plane_point = plane[:, 1:4]
        normal = plane[:, -3:]
        plane_tree = KDTree(plane_point*self.celllength)
        dd, ii = plane_tree.query(pos, k=1)
        plane_point_int = np.array(plane_point[ii]).astype(int)
        
        plane_vaccum_tree = KDTree(plane_vaccum[:, 1:4]*self.celllength)
        dd_v, ii_v = plane_vaccum_tree.query(pos, k=1)
        plane_point_vaccum_int = np.array(plane_vaccum[ii_v, 1:4]).astype(int)

        return plane_point_int, normal[ii], plane_point_vaccum_int

    def get_inject_theta(self, plane, pos, vel):
        # plane = self.get_pointcloud(film)
        plane_point = plane[:, 3:6]
        normal = plane[:, :3]
        velocity_normal = np.linalg.norm(vel, axis=1)
        velocity = np.divide(vel.T, velocity_normal).T
        plane_tree = KDTree(plane_point*self.celllength)

        dd, ii = plane_tree.query(pos, k=1)

        dot_products = np.einsum('...i,...i->...', velocity, normal[ii])
        theta = np.arccos(dot_products)
        return theta
    
    def get_yield(self, theta):
        # yield_func = interpolate.interp1d(self.yield_hist[1], self.yield_hist[0], kind='quadratic')
        etch_yield = self.yield_func(theta)
        return etch_yield
