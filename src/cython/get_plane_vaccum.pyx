# 文件名：get_plane_vaccum.pyx
import numpy as np
cimport numpy as cnp
import cython

@cython.boundscheck(False)  # 关闭边界检查
@cython.wraparound(False)   # 关闭负索引支持
def get_plane_vaccum(
    cnp.ndarray[cnp.int_t, ndim=2] get_plane,
    cnp.ndarray[cnp.double_t, ndim=4] film_label_index_normal_mirror,
    int mirrorGap
):

    cdef int n = get_plane.shape[0]
    cdef int i, candi
    cdef cnp.ndarray[cnp.int_t, ndim=2] get_plane_vaccum = np.zeros((n, 3), dtype=np.int32)
    cdef cnp.ndarray[cnp.int_t, ndim=2] grid_cross = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=np.int32)
    cdef cnp.ndarray[cnp.int_t, ndim=2] point_nn
    cdef cnp.ndarray[cnp.npy_bool, ndim=1] mask
    cdef cnp.ndarray[cnp.int_t, ndim=2] vaccum_candidate

    # 遍历每个点
    for i in range(n):
        
        # 处理真空点
        point_nn = grid_cross + get_plane[i]
        point_nn[:, 0] += mirrorGap
        point_nn[:, 1] += mirrorGap

        # 筛选邻居点 film_label 值为 -1 的点
        mask = film_label_index_normal_mirror[point_nn[:, 0], point_nn[:, 1], point_nn[:, 2], 0] == -1

        # 获取候选点
        vaccum_candidate = film_label_index_normal_mirror[point_nn[mask, 0], point_nn[mask, 1], point_nn[mask, 2], 1:4].astype(np.int32)

        if vaccum_candidate.shape[0] > 0:  # 避免空候选
            candi = np.random.randint(vaccum_candidate.shape[0])
            get_plane_vaccum[i, :] = vaccum_candidate[candi, :]

    return get_plane_vaccum


