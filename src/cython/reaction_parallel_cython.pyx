# distutils: language = c++


import numpy as np
cimport cython
from cython.parallel cimport prange, parallel
cimport numpy as cnp
from libc.math cimport fabs, acos
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp.random cimport random_device, mt19937, uniform_real_distribution, normal_distribution
from libcpp.vector cimport vector
from libc.math cimport ceil, log, exp, cos, pi, sqrt
from cython.operator cimport dereference as deref


# 创建随机数生成器和分布器
cdef mt19937 rng = mt19937(42)  # 使用种子值 42 初始化随机数生成器
cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0, 1.0)
cdef normal_distribution[double] distn =  normal_distribution[double](0, 1)

cdef int FILMSIZE = 5

cdef struct Particle:
    double[3] pos
    double[3] vel
    double E
    int id


cdef struct Cell:
    int id
    int[3] index
    int[5] film
    double[3] normal



cdef int[2][5][5] react_table_equation = [
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
]


cdef int[3][5] react_type_table = [[1, 1, 1, 4, 0], # 1: chlorination  # 0: no reaction  # 4: Themal etch
                                   [2, 2, 2, 2, 2], # 2 for physics and chemical sputtering
                                   [3, 3, 3, 3, 3]]




# cdef double[5] react_prob_chemical = [0.50, 0.50, 0.50, 0.9, 0.0]
cdef double[5] react_prob_chemical = [1.0, 0.0, 0.0, 0.0, 0.0]

cdef double[5] react_yield_p0 = [0.30, 0.30, 0.30, 0.30, 0.30]

# cdef double[4][4] rn_coeffcients = [[0.9423, 0.9434, 2.742, 3.026], # 100
#                                     [0.9620, 0.9608, 2.542, 3.720],  # 1000
#                                     [0.9458, 0.9445, 2.551, 3.735],  # 1050
#                                     [1.046, 1.046, 2.686, 4.301]]     # 10000

rn_coeffcients = np.array([
    [0.9423, 0.9434, 2.742, 3.026],
    [0.9620, 0.9608, 2.542, 3.720],
    [0.9458, 0.9445, 2.551, 3.735],
    [1.046, 1.046, 2.686, 4.301]
], dtype=np.double)


def Rn_coeffcient(double c1, 
                  double c2, 
                  double c3, 
                  double c4, 
                  double alpha):
    cdef double rn_prob
    rn_prob = c1 + c2*np.tanh(c3*alpha - c4)
    return rn_prob

def Rn_probability(double[:] c_list):

    rn_angle = np.linspace(0, np.pi/2, 180, dtype=np.double)
    rn_angle_view: cython.double[:] = rn_angle

    i_max: cython.Py_ssize_t = rn_angle.shape[0]
    i: cython.Py_ssize_t

    rn_prob = np.zeros(i_max, dtype=np.double)
    rn_prob_view: cython.double[:] = rn_prob

    for i in range(i_max):
        rn_prob_view[i] = Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], rn_angle_view[i])
    # rn_prob = [Rn_coeffcient(c_list[0], c_list[1], c_list[2], c_list[3], i) for i in rn_angle]
    for i in range(i_max):
        rn_prob_view[i] /= rn_prob_view[i_max-1]
    for i in range(i_max):
        rn_prob_view[i] = 1-rn_prob_view[i]
    return rn_prob


# def Rn_matrix_func(double[:,:] rn_coeffcients):

cdef int p, pp



# return rn_matrix

cdef double[4][180] rn_matrix 

for p in range(rn_coeffcients.shape[0]):
    rn_prob = Rn_probability(rn_coeffcients[p])
    for pp in range(180):
        rn_matrix[p][pp] = rn_prob[pp]



cdef double[4] rn_energy = [100, 1000, 1050, 10000]
# cdef double[180] rn_angle 



# 使用 C 风格数组
cdef double[180] rn_angle
np_angle = np.linspace(0, np.pi / 2, 180, dtype=np.double)

# 逐一赋值到 C 数组
for i in range(180):
    rn_angle[i] = np_angle[i]

cdef double[5] react_redepo_sticking = [1.0, 1.0, 1.0, 1.0, 1.0]

@cython.boundscheck(False)  # 禁用边界检查以提升性能
@cython.wraparound(False)   # 禁用负索引支持以提升性能
cdef double linear_interp(double x, double* xp, double* fp) noexcept nogil:
    """
    C 实现的线性插值函数
    :param x: 待插值的值
    :param xp: x 轴的已知点数组
    :param fp: y 轴的已知点数组
    :param n: xp 和 fp 的长度
    :return: 插值后的值
    """
    cdef int i
    cdef double x0, x1, y0, y1

    # 检查边界情况
    if x <= xp[0]:
        return fp[0]
    if x >= xp[179]:
        return fp[179]

    # 找到 x 所在的区间 [xp[i], xp[i+1]]
    for i in range(179):
        if xp[i] <= x < xp[i + 1]:
            x0 = xp[i]
            x1 = xp[i + 1]
            y0 = fp[i]
            y1 = fp[i + 1]
            # 线性插值公式
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    # 如果没有找到区间（理论上不应该发生），返回 0
    return 0.0

# # 通过 C 标准库 rand 生成 [0, 1) 之间的随机数
# cdef double generate_random_number() noexcept nogil:
#     return rand() / RAND_MAX






@cython.boundscheck(False)  # 禁用边界检查以提升性能
@cython.wraparound(False)   # 禁用负索引支持以提升性能
cdef int* sticking_probability_structed(Particle particle, Cell cell, double angle_rad) noexcept nogil:

    cdef int i
    cdef int j
    # choice = np.random.rand(5)
    # choice_view: cython.double[:] = choice
    # 使用 C 标准库生成随机数
    cdef double[5] choice
    cdef int* sticking_acceptList = <int*>malloc(5 * sizeof(int))  # 动态分配内存

    for i in range(FILMSIZE):
        choice[i] = dist(rng)  # 用 C 库的随机数填充数组
    # cdef bint[5] sticking_acceptList

    cdef int e, energy_range
    cdef double sticking_rate = 0
    
    if particle.id >= 2:
        particle.id = 2

    for i in range(FILMSIZE):
        sticking_acceptList[i] = 0
        if cell.film[i] <= 0:
            choice[i] = 1.0

    for j in range(FILMSIZE):
        if particle.id == 1:  
            energy_range = 0
            for e in range(FILMSIZE):
                if particle.E < rn_energy[e]:
                    energy_range = e
                    break
            sticking_rate = linear_interp(angle_rad, &rn_angle[0], &rn_matrix[energy_range][0])
        elif particle.id == 0:  # 化学反应
            sticking_rate = react_prob_chemical[j]
        elif particle.id >= 2:  # redepo
            sticking_rate = react_redepo_sticking[particle.id - 2]

        if sticking_rate > choice[j]:
            sticking_acceptList[j] = 1

    return sticking_acceptList


cdef double dot3(double[3] a, double[3] b) noexcept nogil:
    cdef double result = 0
    cdef int i
    for i in range(3):
        result += a[i]*b[i]
    return result






@cython.boundscheck(False)  # 禁用边界检查以提升性能
@cython.wraparound(False)   # 禁用负索引支持以提升性能
cdef int* find_max_position(vector[double]* arr) noexcept nogil:
    """
    找出 C 数组 double[5] 中最大值的位置
    """
    cdef int i
    cdef int* max_pos = <int*>malloc(sizeof(int))
    cdef double max_val = 0  # 初始化最大值为数组的第一个元素

    # 遍历数组
    max_pos[0] = -1
    for i in range(FILMSIZE):  # 从第二个元素开始遍历
        if deref(arr)[i] > max_val:
            max_val = deref(arr)[i]
            max_pos[0] = i

    return max_pos

cdef void film_add(int[5] film, int* react_add) noexcept nogil:
    cdef int i = 0
    for i in range(FILMSIZE):
        film[i] += react_add[i]


cdef int film_sum(int[5] film) noexcept nogil:
    cdef int i
    cdef int sum = 0
    for i in range(FILMSIZE):
        sum += film[i]
    return sum



cdef int* react_add_func(int id, int choice) noexcept nogil:
    cdef int *react_add
    cdef int i
    react_add = <int*>malloc(FILMSIZE * sizeof(int))
    for i in range(FILMSIZE):
        # if id <= 1: # gas Cl Ion
        react_add[i] = react_table_equation[id][choice][i]
        # else:

    return react_add



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sputter_yield_angle(double gamma0, double gammaMax, double thetaMax, double[2][180] sputterYield_ion):
    """
    Calculate sputter yield as a function of angle.
    """
    cdef double f = -log(gammaMax / gamma0) / (log(cos(gammaMax)) + 1 - cos(thetaMax))
    cdef double s = f * cos(thetaMax)
    cdef int n = 180
    cdef cnp.ndarray[cnp.double_t, ndim=1] theta = np.linspace(0, pi / 2, n, dtype=np.float64)
    cdef double[:] theta_view = theta
    # cdef cnp.ndarray[cnp.double_t, ndim=1] sputterYield = np.zeros(n, dtype=np.float64)
    cdef double[180] sputterYield
    # cdef double[2][180] yield_hist 

    cdef int i
    for i in range(n):
        if i == n-1:
            sputterYield[i] = 0
        else:
            sputterYield[i] = gamma0 * cos(theta_view[i])**(-f) * exp(-s * (1 / cos(theta_view[i]) - 1))
    
        sputterYield_ion[0][i] = sputterYield[i]
        sputterYield_ion[1][i] = theta[i]

    # return yield_hist

cdef double[2][180] sputterYield_ion

sputter_yield_angle(0.3, 0.001, pi/4, sputterYield_ion)



# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double sputter_yield(double p0, double theta, double energy, double Eth) noexcept nogil:
#     """
#     Combined sputter yield calculation.
#     """
#     cdef double interp_value
#     cdef int i
#     cdef double react_yield
#     # Interpolate sputter yield at a given angle
#     interp_value = 0  # Handle out-of-bounds cases
#     for i in range(179):
#         if sputterYield_ion[1][i] <= theta < sputterYield_ion[1][i + 1]:
#             interp_value = sputterYield_ion[0][i] + (sputterYield_ion[0][i + 1] - sputterYield_ion[0][i]) * \
#                            (theta - sputterYield_ion[1][i]) / (sputterYield_ion[1][i + 1] - sputterYield_ion[1][i])
#             break

#     react_yield = p0 * interp_value * (energy**0.5 - Eth**0.5)
#     return react_yield

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double sputter_yield(double p0, double theta, double energy, double Eth) noexcept nogil:
    """
    Combined sputter yield calculation.
    """
    cdef double interp_value = 0.0
    cdef int i
    cdef double sqrt_energy, sqrt_Eth, diff, product

    # Interpolate sputter yield at a given angle
    for i in range(179):
        if sputterYield_ion[1][i] <= theta < sputterYield_ion[1][i + 1]:
            interp_value = sputterYield_ion[0][i] + (sputterYield_ion[0][i + 1] - sputterYield_ion[0][i]) * \
                           (theta - sputterYield_ion[1][i]) / (sputterYield_ion[1][i + 1] - sputterYield_ion[1][i])
            break

    # Calculate react_yield
    sqrt_energy = energy**0.5
    sqrt_Eth = Eth**0.5
    diff = sqrt_energy - sqrt_Eth
    product = p0 * interp_value * diff

    return product

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def sputter_yield_p(double p0, double theta, double energy, double Eth):
#     """
#     Combined sputter yield calculation.
#     """
#     cdef double interp_value
#     cdef int i
#     cdef double react_yield
#     # Interpolate sputter yield at a given angle
#     interp_value = 0  # Handle out-of-bounds cases
#     for i in range(179):
#         if sputterYield_ion[1][i] <= theta < sputterYield_ion[1][i + 1]:
#             interp_value = sputterYield_ion[0][i] + (sputterYield_ion[0][i + 1] - sputterYield_ion[0][i]) * \
#                            (theta - sputterYield_ion[1][i]) / (sputterYield_ion[1][i + 1] - sputterYield_ion[1][i])
#             break

#     react_yield = p0 * interp_value * (energy**0.5 - Eth**0.5)
#     return react_yield

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void SpecularReflect(double[3] vel, double[3] normal) noexcept nogil:
    # 计算速度和法向量的点积
    cdef double dot_product_2 = dot3(vel, normal)*2
    cdef int i
    # 计算反射速度
    for i in range(3):
        vel[i] -= dot_product_2 * normal[i]

cdef double normalizer(double[3] a) noexcept nogil:
    cdef double result
    cdef double temp = 0
    cdef int i
    for i in range(3):
        temp += a[i]**2
    result = sqrt(temp)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void DiffusionReflect(double[3] vel, double[3] normal, double randn1, double randn2, double rand3,) noexcept nogil:
    """
    Compute the diffusion reflection of a particle velocity vector off a surface.
    :param vel: Particle velocity vector
    :param normal: Surface normal vector
    :param UN: Output normalized reflected velocity vector
    """
    cdef double dot_product, norm_Ut, pm, norm_U
    cdef double[3] Ut, tw1, tw2, U
    
    # Calculate the tangential component Ut
    dot_product = dot3(vel, normal)
    for i in range(3):
        Ut[i] = vel[i] - dot_product * normal[i]
    
    # Normalize Ut to get tw1
    # norm_Ut = sqrt(Ut[0]**2 + Ut[1]**2 + Ut[2]**2)
    norm_Ut = normalizer(Ut)
    for i in range(3):
        tw1[i] = Ut[i] / norm_Ut

    # Calculate tw2 as the cross product of tw1 and normal
    tw2[0] = tw1[1] * normal[2] - tw1[2] * normal[1]
    tw2[1] = tw1[2] * normal[0] - tw1[0] * normal[2]
    tw2[2] = tw1[0] * normal[1] - tw1[1] * normal[0]

    # Determine the sign based on the dot product
    if dot_product > 0:
        pm = -1.0
    else:
        pm = 1.0

    # Generate random values for the new velocity
    # rand1 = normal_distribution[double](0, 1)(mt19937(random_device()()))
    # rand2 = normal_distribution[double](0, 1)(mt19937(random_device()()))
    # rand3 = -2.0 * log(1.0 - normal_distribution[double](0, 1)(mt19937(random_device()())))

    for i in range(3):
        U[i] = randn1 * tw1[i] + randn2 * tw2[i] + pm * sqrt(-2.0 * log(1.0 -rand3)) * normal[i]

    # Normalize U to get UN
    # norm_U = sqrt(U[0]**2 + U[1]**2 + U[2]**2)
    norm_U = normalizer(U)
    for i in range(3):
        vel[i] = U[i] / norm_U






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void MoveParticle(double[3] vel, double[3] pos) noexcept nogil:
    cdef int i
    for i in range(3):
        pos[i] += vel[i]







# 粒子-网格反应主函数
@cython.boundscheck(False)  # 禁用边界检查以提升性能
@cython.wraparound(False)   # 禁用负索引支持以提升性能
def particle_parallel_vector(Particle[:] particles, Cell[:,:,:] cell, double[:] cellSizeXYZ):

    cdef Py_ssize_t i, j, k
    cdef mt19937 local_rng
    cdef random_device rd
    cdef int *celli
    cdef int *cellj
    cdef int *cellk
    cdef double *angle_rad
    cdef double *dot_product
    cdef vector[double] *react_choice_random
    cdef int *react_choice
    cdef int *react_add
    cdef int *sticking_acceptList
    cdef int *react_type
    cdef double *react_yield
    cdef double *randn1
    cdef double *randn2
    cdef double *rand3
    # cdef int react_type
    # cdef vector[int] update_film_etch
    cdef int particleCount = particles.shape[0]

    # particleCount = particles.shape[0]
    sticking_acceptList_indices = np.zeros((particles.shape[0], 5), dtype=np.int32)
    cdef int[:, :] sticking_acceptList_indices_view = sticking_acceptList_indices

    react_choice_random_indices = np.zeros((particles.shape[0], 5), dtype=np.double)
    cdef double[:, :] react_choice_random_indices_view = react_choice_random_indices

    dot_product_all = np.zeros(particles.shape[0], dtype=np.double)
    cdef double[:] dot_product_view = dot_product_all

    react_choice_indices = np.zeros(particles.shape[0], dtype=np.int32)
    cdef int[:] react_choice_indices_view = react_choice_indices

    bool_mask = np.ones(particles.shape[0], dtype=np.int32)
    cdef int[:] bool_mask_view = bool_mask

    react_type_indices = np.zeros(particles.shape[0], dtype=np.int32)
    cdef int[:] react_type_indices_view = react_type_indices

    react_add_indices = np.zeros((particles.shape[0], 5), dtype=np.int32)
    cdef int[:, :] react_add_indices_view = react_add_indices

    update_film_etch = np.ones((particles.shape[0], 3), dtype=np.int32)*-1
    cdef int[:, :] update_film_etch_view = update_film_etch

    # 遍历所有粒子
    with nogil, parallel():
        celli = <int*>malloc(sizeof(int))
        cellj = <int*>malloc(sizeof(int))
        cellk = <int*>malloc(sizeof(int))
        angle_rad = <double*>malloc(sizeof(double))
        dot_product = <double*>malloc(sizeof(double))
        react_type = <int*>malloc(sizeof(int))
        # react_choice = <int*>malloc(sizeof(int))
        react_choice_random = new vector[double](FILMSIZE)
        react_yield = <double*>malloc(sizeof(double))

        randn1 = <double*>malloc(sizeof(double))
        randn2 = <double*>malloc(sizeof(double))
        rand3 = <double*>malloc(sizeof(double))
        try:
            for i in prange(particleCount):
                local_rng = mt19937(rd() + i)  # 通过线程ID i生成唯一的种子
            # for i in range(particleCount):
                celli[0] = <int> particles[i].pos[0]
                cellj[0] = <int> particles[i].pos[1]
                cellk[0] = <int> particles[i].pos[2]

                if cell[celli[0], cellj[0], cellk[0]].id == 1:
                    dot_product[0] = dot3(particles[i].vel, cell[celli[0], cellj[0], cellk[0]].normal)
                    dot_product[0] = fabs(dot_product[0])
                    angle_rad[0] = acos(dot_product[0])
                    dot_product_view[i] = angle_rad[0] # view
                    sticking_acceptList = sticking_probability_structed(particles[i], cell[celli[0], cellj[0], cellk[0]], angle_rad[0])

                    for j in range(FILMSIZE):
                        sticking_acceptList_indices_view[i, j] = sticking_acceptList[j]  # view
                        if sticking_acceptList[j] == 1:
                            deref(react_choice_random)[j] = dist(local_rng)
                        else:
                            deref(react_choice_random)[j] = 0
                        react_choice_random_indices_view[i, j] = deref(react_choice_random)[j]  # view
                    react_choice = find_max_position(react_choice_random)
                    react_choice_indices_view[i] = react_choice[0]  # view
                    if react_choice[0] != -1:
                        bool_mask_view[i] = 0
                    react_type[0] = react_type_table[particles[i].id][react_choice[0]]
                    react_type_indices_view[i] = react_type[0]  # view

                    react_add = react_add_func(particles[i].id, react_choice[0])
                    for k in range(FILMSIZE):
                        react_add_indices_view[i, k] = react_add[k]
                    if react_type[0] == 1: # chemical transfer
                        film_add(cell[celli[0], cellj[0], cellk[0]].film, react_add)
                    elif react_type[0] == 4: # chemical remove
                        film_add(cell[celli[0], cellj[0], cellk[0]].film, react_add)
                        if film_sum(cell[celli[0], cellj[0], cellk[0]].film) == 0:
                            update_film_etch_view[i,0] = celli[0]  # view
                            update_film_etch_view[i,1] = cellj[0]  # view
                            update_film_etch_view[i,2] = cellk[0]  # view
                    elif react_type[0] == 2: # physics sputter
                        react_yield[0] = sputter_yield(react_yield_p0[0], angle_rad[0], particles[i].E, 5)
                        if react_yield[0] > dist(local_rng):
                            film_add(cell[celli[0], cellj[0], cellk[0]].film, react_add)
                            if film_sum(cell[celli[0], cellj[0], cellk[0]].film) == 0:
                                update_film_etch_view[i,0] = celli[0]  # view
                                update_film_etch_view[i,1] = cellj[0]  # view
                                update_film_etch_view[i,2] = cellk[0]  # view
                    if react_choice[0] == -1: # reflection
                        # SpecularReflect(particles[i].vel, cell[celli[0], cellj[0], cellk[0]].normal)
                        randn1[0] = distn(local_rng)
                        randn2[0] = distn(local_rng)
                        rand3[0] = dist(local_rng)
                        DiffusionReflect(particles[i].vel, cell[celli[0], cellj[0], cellk[0]].normal, randn1[0], randn2[0], rand3[0])

                    free(sticking_acceptList)
                    free(react_choice)
                    free(react_add)

                MoveParticle(particles[i].vel, particles[i].pos)

                if bool_mask_view[i] == 1: # boundary
                    if particles[i].pos[0] >= cellSizeXYZ[0]:
                        particles[i].pos[0] -= cellSizeXYZ[0]
                    elif particles[i].pos[0] < 0:
                        particles[i].pos[0] += cellSizeXYZ[0]
                    if particles[i].pos[1] >= cellSizeXYZ[1]:
                        particles[i].pos[1] -= cellSizeXYZ[1]
                    elif particles[i].pos[1] < 0:
                        particles[i].pos[1] += cellSizeXYZ[1]
                    if (particles[i].pos[0] > cellSizeXYZ[0] or particles[i].pos[0] < 0 or
                        particles[i].pos[1] > cellSizeXYZ[1] or particles[i].pos[1] < 0 or
                        particles[i].pos[2] > cellSizeXYZ[2] or particles[i].pos[2] < 0):
                        bool_mask_view[i] = 0

        finally:
            del react_choice_random
            # del react_add
            free(celli)
            free(cellj)
            free(cellk)
            free(angle_rad)
            free(dot_product)
            free(react_type)
            free(react_yield)
            free(randn1)
            free(randn2)
            free(rand3)
    # return dot_product_all
    # return sticking_acceptList_indices
    return react_choice_indices, \
            sticking_acceptList_indices, \
            react_choice_random_indices, \
            react_type_indices, \
            react_add_indices,\
            update_film_etch,\
            bool_mask



