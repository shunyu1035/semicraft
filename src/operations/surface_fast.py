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
        point[0] -= mirrorGap
        point[1] -= mirrorGap
        normal_matrix[point[0], point[1], point[2]] = normal
        return normal_matrix

    def build_normal_matrix(self, film, mirrorGap):
        surface_film = self.scanZ_numpy(film)
        vacuum_film = self.scanZ_vacuum_numpy(film)
        cellSizeX, cellSizeY, cellSizeZ = surface_film.shape

        surface_mirror = np.zeros((cellSizeX+int(mirrorGap*2), cellSizeY+int(mirrorGap*2), cellSizeZ))
        film_mirror = mirror.update_surface_mirror(surface_film, surface_mirror, mirrorGap, cellSizeX, cellSizeY)

        # normal_array shape(x,y,z,3) to film shape(x,y,z)
        normal_matrix = np.zeros((cellSizeX, cellSizeY, cellSizeZ, 3))
        surface_point = np.array(np.where(surface_film == 1)).T

        for i in range(surface_point.shape[0]):
            normal_matrix = self.get_normal_from_grid(film_mirror, normal_matrix, mirrorGap, surface_point[i])
        return normal_matrix, vacuum_film

    def update_normal_matrix(self, film_mirror, normal_matrix, mirrorGap, point_to_change):
        point_nn_all = np.zeros((1,3))
        for i in range(point_to_change.shape[0]):
            x, y, z = point_to_change[i]
            x += mirrorGap
            y += mirrorGap
            # print(x)
            grid_cube = film_mirror[x-3:x+4, y-3:y+4, z-3:z+4]
            # print(grid_cube)
            point_nn = np.array(np.where(grid_cube == 1)).T
            point_nn += point_to_change[i]
            point_nn_all = np.vstack((point_nn_all, point_nn))
        point_nn_all = np.unique(point_nn_all[1:], axis=0).astype(np.int64)
        # print(point_nn_all)
        for i in range(point_nn_all.shape[0]):
            normal_matrix = self.get_normal_from_grid(film_mirror, normal_matrix, mirrorGap, point_nn_all[i])
        return normal_matrix




    def get_inject_normal(self, plane, plane_vaccum, pos, vel):
        # plane = self.get_pointcloud(film)
        plane_point = plane[:, 3:6]
        normal = plane[:, :3]
        velocity_normal = np.linalg.norm(vel, axis=1)
        velocity = np.divide(vel.T, velocity_normal).T
        plane_tree = KDTree(plane_point*self.celllength)

        dd, ii = plane_tree.query(pos, k=1)
        plane_point_int = np.array(plane_point[ii]).astype(int)
        # dot_products = np.einsum('...i,...i->...', velocity, normal[ii])
        # theta = np.arccos(dot_products)
        plane_vaccum_tree = KDTree(plane_vaccum*self.celllength)
        dd_v, ii_v = plane_vaccum_tree.query(pos, k=1)
        plane_point_vaccum_int = np.array(plane_vaccum[ii_v]).astype(int)

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

    # def scanZ_numpy(self,film):
    #     # 初始化一个全零的表面数组
    #     surface_sparse_depo = np.zeros_like(film)

    #     # depo 条件
    #     current_plane = film != 0

    #     # 创建邻居布尔索引数组
    #     neighbors = np.zeros_like(film, dtype=bool)

    #     # 获取周围邻居的布尔索引
    #     neighbors[1:, :, :] |= film[:-1, :, :] == 0  # 上面
    #     neighbors[:-1, :, :] |= film[1:, :, :] == 0  # 下面
    #     neighbors[:, 1:, :] |= film[:, :-1, :] == 0  # 左边
    #     neighbors[:, :-1, :] |= film[:, 1:, :] == 0  # 右边
    #     neighbors[:, :, 1:] |= film[:, :, :-1] == 0  # 前面
    #     neighbors[:, :, :-1] |= film[:, :, 1:] == 0  # 后面

    #     # 获取满足条件的索引
    #     condition_depo = current_plane & neighbors

    #     # 更新表面稀疏张量
    #     surface_sparse_depo[condition_depo] = 1

    #     return surface_sparse_depo
            
    # def scanZ_vaccum(self, film): # fast scanZ
    #     # 初始化一个全零的表面数组
    #     surface_sparse = np.zeros_like(film)

    #     # depo 条件
    #     current_plane = film == 0

    #     # 创建邻居布尔索引数组
    #     neighbors = np.zeros_like(film, dtype=bool)
        
    #     neighbors[1:, :, :] |= film[:-1, :, :] > 0  # 上面
    #     neighbors[:-1, :, :] |= film[1:, :, :] > 0  # 下面
    #     neighbors[:, 1:, :] |= film[:, :-1, :] > 0  # 左边
    #     neighbors[:, :-1, :] |= film[:, 1:, :] > 0  # 右边
    #     neighbors[:, :, 1:] |= film[:, :, :-1] > 0  # 前面
    #     neighbors[:, :, :-1] |= film[:, :, 1:] > 0  # 后面
        
    #     # 获取满足条件的索引
    #     condition = current_plane & neighbors
        
    #     # 更新表面稀疏张量
    #     surface_sparse[condition] = 1
        

    #     return surface_sparse