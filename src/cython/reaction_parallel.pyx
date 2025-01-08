# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp
from libc.math cimport floor

# 声明全局变量
cdef cnp.ndarray[cnp.float64_t, ndim=2] rn_matrix
cdef cnp.ndarray[cnp.float64_t, ndim=1] rn_energy, rn_angle
cdef cnp.ndarray[cnp.float64_t, ndim=1] react_prob_chemical, react_redepo_sticking

# sticking_probability 函数
def sticking_probability(cnp.ndarray[cnp.float64_t, ndim=1] parcel, 
                         cnp.ndarray[cnp.int32_t, ndim=1] film, 
                         double angle_rad):
    cdef int film_layer = film.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] choice = np.random.rand(film_layer)
    cdef cnp.ndarray[cnp.bool_t, ndim=1] sticking_acceptList = np.zeros(film_layer, dtype=np.bool_)
    cdef int particle = int(parcel[-1])
    cdef int j, e, energy_range
    cdef double sticking_rate

    if particle >= 2:
        particle = 2

    for j in range(film_layer):
        if film[j] <= 0:
            choice[j] = 1.0

    for j in range(film_layer):
        if particle == 1:  # 离子溅射
            energy_range = 0
            for e in range(rn_energy.shape[0]):
                if parcel[-2] < rn_energy[e]:
                    energy_range = e
                    break
            sticking_rate = np.interp(angle_rad, rn_angle, rn_matrix[energy_range])
        elif particle == 0:  # 化学反应
            sticking_rate = react_prob_chemical[j]
        elif particle >= 2:  # redepo
            sticking_rate = react_redepo_sticking[int(parcel[-1]) - 2]

        if sticking_rate > choice[j]:
            sticking_acceptList[j] = True

    return sticking_acceptList, particle

# reaction_rate_parallel_all 函数
def reaction_rate_parallel_all(cnp.ndarray[cnp.int32_t, ndim=4] filmMatrix, 
                               cnp.ndarray[cnp.float64_t, ndim=2] parcel, 
                               cnp.ndarray[cnp.float64_t, ndim=4] film_label_index_normal):
    cdef int elements = filmMatrix.shape[-1]
    cdef int i, film_layer
    cdef cnp.ndarray[cnp.bool_t, ndim=3] update_film_etch = np.zeros(filmMatrix.shape[:3], dtype=np.bool_)
    cdef cnp.ndarray[cnp.bool_t, ndim=3] update_film_depo = np.zeros(filmMatrix.shape[:3], dtype=np.bool_)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] reactList = np.ones(parcel.shape[0], dtype=np.int32) * -1
    cdef cnp.ndarray[cnp.bool_t, ndim=1] indice_1 = np.ones(parcel.shape[0], dtype=np.bool_)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] depo_parcel = np.zeros(parcel.shape[0])

    for i in prange(parcel.shape[0], nogil=True):
        # 包含其余的计算逻辑
        pass

    return filmMatrix, parcel, update_film_etch, update_film_depo, reactList, depo_parcel
