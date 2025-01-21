# distutils: language = c++


import numpy as np
cimport cython
from cython.parallel cimport prange, parallel
cimport numpy as cnp
from libc.math cimport fabs, acos
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp.random cimport mt19937, uniform_real_distribution, normal_distribution
from libcpp.vector cimport vector
from libc.math cimport ceil
from cython.operator cimport dereference as deref


# 创建随机数生成器和分布器
cdef mt19937 rng = mt19937(42)  # 使用种子值 42 初始化随机数生成器
cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0, 1.0)


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



def react_add_test():
    cdef int[5] test
    cdef int i
    testpy = np.zeros(FILMSIZE, dtype=np.int32)  # 确保类型匹配
    test = react_table_equation[1][4]  # 确保 react_table_equation 是 cdef 定义的二维数组
    for i in range(FILMSIZE):
        testpy[i] = test[i]  # 将 C 数组的值赋给 NumPy 数组
        print(testpy[i])  # 打印每个值
    return testpy  # 返回 NumPy 数组




def react_add_test2(Particle particle):
    cdef int[5] test

    testpy = np.zeros(5, dtype=np.int32)  # 确保类型匹配
    test = react_table_equation[particle.id][1]  # 确保 react_table_equation 是 cdef 定义的二维数组

    testpy = test  # 将 C 数组的值赋给 NumPy 数组
    print(testpy)  # 打印每个值
    return testpy  # 返回 NumPy 数组


def react_add_test3(Particle[:] particle):
    cdef int[5] test  # Use a fixed size array
    cdef int i, j, k
    cdef cnp.ndarray[cnp.int32_t, ndim=2] testpy = np.zeros((particle.shape[0], 5), dtype=np.int32)  # NumPy array
    cdef int[:, :] testpy_view = testpy

    # Parallelize the loop with prange
    for i in prange(particle.shape[0], nogil=True):
        # Load data into C array
        for k in range(5):
            test[k] = react_table_equation[particle[i].id][1][k]

        # Assign values back to NumPy array
        for j in range(5):
            testpy_view[i, j] = test[j]
        
    return testpy  # Return the result as a NumPy array



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


def pyprint(data):
    datapy = np.zeros_like(data)
    datapy = data
    print(datapy)

# pyprint(rn_matrix)
# rn_matrix = np.array([Rn_probability(i) for i in rn_coeffcients])




cdef double[4] rn_energy = [100, 1000, 1050, 10000]
# cdef double[180] rn_angle 

@cython.boundscheck(False)  # 禁用边界检查
@cython.wraparound(False)   # 禁用负索引支持
def rn_angle_np():

    cdef double[180] rn_angle
    # 用 NumPy 生成数据
    np_angle = np.linspace(0, np.pi / 2, 180, dtype=np.double)
    
    # 逐一赋值到 C 数组
    for i in range(180):
        rn_angle[i] = np_angle[i]
    return rn_angle

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

cdef int* film_add(int[5] a, int[5] b) noexcept nogil:
    cdef int i = 0
    cdef int *film
    film = <int*>malloc(FILMSIZE * sizeof(int))
    for i in range(5):
        film[i] = a[i] + b[i]
    return film


# cdef double* double5_add(double[5] a, double[5] b) noexcept nogil:
#     cdef int i = 0
#     for i in range(5):
#         a[i] += b[i]
#     return a


def vector_to_numpy_2d():
    cdef vector[vector[int]] cpp_2d_array
    cdef int rows = 10, cols = 4
    cdef int i, j, k, d
    # cdef vector[int] ptr

    # ptr.resize(cols)
    # 创建一个 3x4 的 C++ 二维 vector 并赋值
    cpp_2d_array.resize(rows)
    # for i in range(rows):
    #     cpp_2d_array[i].resize(cols)

    # for i in range(rows):
    for i in prange(rows, nogil=True):
        cpp_2d_array[i].resize(cols)
        # cdef vector[int] ptr
        # ptr.resize(cols)
        for j in range(cols):
            cpp_2d_array[i][j] = i * cols + j  # 示例赋值：每个元素 = 行号 * 列数 + 列号
            # ptr[j] = i * cols + j
        # cpp_2d_array[i] = ptr  # 示例赋值：每个元素 = 行号 * 列数 + 列号
        # ptr.resize(cols)
        # for j in range(cols):
        #     ptr[j] = 0

    # 创建一个与 C++ 二维数组形状相同的 NumPy 数组
    cdef cnp.ndarray[cnp.int32_t, ndim=2] np_2d_array = np.zeros((rows, cols), dtype=np.int32)

    # 将 C++ vector 中的值复制到 NumPy 数组
    for k in range(rows):
        for d in range(cols):
            np_2d_array[k, d] = cpp_2d_array[k][d]

    return np_2d_array


cdef int* react_add_func(int id, int choice) noexcept nogil:
    cdef int *react_add
    cdef int i
    react_add = <int*>malloc(FILMSIZE * sizeof(int))
    for i in range(FILMSIZE):
        # if id <= 1: # gas Cl Ion
        react_add[i] = react_table_equation[id][choice][i]
        # else:

    return react_add



@cython.boundscheck(False)  # 禁用边界检查以提升性能
@cython.wraparound(False)   # 禁用负索引支持以提升性能
def particle_parallel_vector(Particle[:] particles, Cell[:,:,:] cell):

    cdef Py_ssize_t i, j, k
    cdef mt19937 local_rng
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
    # cdef int react_type
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

    react_type_indices = np.zeros(particles.shape[0], dtype=np.int32)
    cdef int[:] react_type_indices_view = react_type_indices

    react_add_indices = np.zeros((particles.shape[0], 5), dtype=np.int32)
    cdef int[:, :] react_add_indices_view = react_add_indices

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

        try:
            for i in prange(particleCount):
                local_rng = mt19937(42 + i)  # 通过线程ID i生成唯一的种子
            # for i in range(particleCount):
                celli[0] = <int> particles[i].pos[0]
                cellj[0] = <int> particles[i].pos[1]
                cellk[0] = <int> particles[i].pos[2]

                if cell[celli[0], cellj[0], cellk[0]].id == 1:
                    dot_product[0] = dot3(particles[i].vel, cell[celli[0], cellj[0], cellk[0]].normal)
                    dot_product[0] = fabs(dot_product[0])
                    angle_rad[0] = acos(dot_product[0])

                    sticking_acceptList = sticking_probability_structed(particles[i], cell[celli[0], cellj[0], cellk[0]], angle_rad[0])

                    for j in range(FILMSIZE):
                        sticking_acceptList_indices_view[i, j] = sticking_acceptList[j]
                        if sticking_acceptList[j] == 1:
                            deref(react_choice_random)[j] = dist(local_rng)
                        else:
                            deref(react_choice_random)[j] = 0
                        react_choice_random_indices_view[i, j] = deref(react_choice_random)[j]
                    react_choice = find_max_position(react_choice_random)
                    react_choice_indices_view[i] = react_choice[0]

                    react_type[0] = react_type_table[particles[i].id][react_choice[0]]
                    react_type_indices_view[i] = react_type[0]

                    react_add = react_add_func(particles[i].id, react_choice[0])
                    for k in range(FILMSIZE):
                        react_add_indices_view[i, k] = react_add[k]

                    
                    free(sticking_acceptList)
                    free(react_choice)
                    free(react_add)


                dot_product_view[i] = angle_rad[0]











        finally:
            del react_choice_random
            # del react_add
            free(celli)
            free(cellj)
            free(cellk)
            free(angle_rad)
            free(dot_product)
            free(react_type)

    # return dot_product_all
    # return sticking_acceptList_indices
    return react_choice_indices, sticking_acceptList_indices, react_choice_random_indices, react_type_indices, react_add_indices




# @cython.boundscheck(False)  # 禁用边界检查以提升性能
# @cython.wraparound(False)   # 禁用负索引支持以提升性能
# def particle_parallel(Particle[:] particles, Cell[:,:,:] cell):

#     cdef int celli, cellj, cellk
#     cdef int i, j
#     cdef double angle_rad
#     cdef double dot_product
#     cdef double[5] react_choice_random
#     cdef int react_choice
#     cdef int[5] react_add
#     cdef int react_type

#     react_choice_indices = np.zeros(particles.shape[0], dtype=np.int32)
#     cdef int[:] react_choice_indices_view = react_choice_indices

#     sticking_acceptList_indices = np.zeros((particles.shape[0], 5), dtype=np.int32)
#     cdef int[:, :] sticking_acceptList_indices_view = sticking_acceptList_indices

#     dot_product_all = np.zeros(particles.shape[0], dtype=np.double)
#     cdef double[:] dot_product_view = dot_product_all

#     # 遍历所有粒子
#     for i in prange(particles.shape[0], nogil=True):
#         celli = <int> particles[i].pos[0]
#         cellj = <int> particles[i].pos[1]
#         cellk = <int> particles[i].pos[2]

#         if cell[celli, cellj, cellk].id == 1:
#             dot_product = dot3(particles[i].vel, cell[celli, cellj, cellk].normal)
#             dot_product = fabs(dot_product)
#             angle_rad = acos(dot_product)

#             # 调用 `sticking_probability_structed`，返回 std::vector
#             sticking_acceptList = sticking_probability_structed(particles[i], cell[celli, cellj, cellk], angle_rad)

#             # 将 `sticking_acceptList` 转换为 numpy 数组
#             for j in range(FILMSIZE):
#                 sticking_acceptList_indices_view[i, j] = sticking_acceptList[j]

#     return sticking_acceptList_indices


            # for j in range(FILMSIZE):
            #     if sticking_acceptList[j] == 1:
            #         react_choice_random[j] = dist(rng)
            #     else:
            #         react_choice_random[j] = 0
            # react_choice = find_max_position(react_choice_random)
            # react_choice_indices_view[i] = react_choice
    # return react_choice_indices
    #         react_type = react_type_table[particles[i].id][react_choice]
    # #             react_type_indices_view[i] = react_type
    # # return react_type_indices

    #             # # Update film based on reaction type
    #         if particles[i].id <= 1:  # Gas Cl Ion
    #             for k in range(5):
    #                 react_add[k] = react_table_equation[particles[i].id][react_choice][k]
                # react_add = int5_add(react_add, react_table_equation[particles[i].id][react_choice])

            # else:  # Redeposited solid
            #     react_add[particles[i].id - 2] = 1
                # if react_type == 1 or react_type == 4:  # Chemical transfer or remove
                #     cell[celli, cellj, cellk].film += react_add
                #     reactList[i] = react_choice
                #     indice_1[i] = False
                # react_add = react_table_equation[0][0][0]
                # react_add = react_table_equation[particles[i].id, react_choice]
            # react_add_indices_view[i] = react_add[0]





    # return sticking_acceptList_indices


                # print(react_add)
                # else:  # Redeposited solid
                #     react_add = np.zeros(elements, dtype=np.int32)
                #     particle_index = particle_view[i].id - 2
                #     react_add[int(particle_index)] = 1

                # if react_type == 1 or react_type == 4:  # Chemical transfer or remove
                #     cell[celli, cellj, cellk].film += react_add
                    # reactList[i] = react_choice
                    # indice_1[i] = False
        #     react_choice_indices = np.where(sticking_acceptList)[0]
        #     if react_choice_indices.size > 0:
        #         # Randomly select a reaction type
        #         react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
        #         reactList[i] = react_choice
        #         indice_1[i] = False

        #         # Determine reaction type and update particle or film
        #         react_type = absorb.react_type_table[particle_view[i].id, react_choice]

        #         if react_type == 1:  # Chemical transfer
        #             depo_particle[i] = 1
        #         elif react_type == 2:  # Physical sputtering
        #             depo_particle[i] = 2
        #         elif react_type == 3:  # Redeposition
        #             depo_particle[i] = 3
        #         elif react_type == 4:  # Thermal etching
        #             depo_particle[i] = 4
        #         elif react_type == 0:  # No reaction
        #             depo_particle[i] = 0

        #         # Update film based on reaction type
        #         if particle_view[i].id <= 1:  # Gas Cl Ion
        #             react_add = absorb.react_table_equation[particle_view[i].id, react_choice, :]
        #         else:  # Redeposited solid
        #             react_add = np.zeros(elements, dtype=np.int32)
        #             particle_index = particle_view[i].id - 2
        #             react_add[int(particle_index)] = 1

        #         if depo_particle[i] == 1 or depo_particle[i] == 4:  # Chemical transfer or remove
        #             film_matrix_view[celli, cellj, cellk] += react_add
        #             if np.sum(film_matrix_view[celli, cellj, cellk]) == 0:
        #                 update_film_etch[celli, cellj, cellk] = True
        #         elif depo_particle[i] == 2:  # Physical sputtering
        #             react_yield = sputter_yield.sputter_yield(
        #                 absorb.react_yield_p0[0], angle_rad, particle_view[i].E, 10
        #             )
        #             if react_yield > np.random.rand():
        #                 film_matrix_view[celli, cellj, cellk] += react_add
        #                 if np.sum(film_matrix_view[celli, cellj, cellk]) == 0:
        #                     update_film_etch[celli, cellj, cellk] = True

        #     # Save film back to memory
        #     # film_matrix_view[celli, cellj, cellk] = film
            
        #     # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
        #     if reactList[i] == -1:
        #         # particle[i, 3:6] = reflect.SpecularReflect(particle[i, 3:6], get_theta[i])
        #         particle[i].vel = reflection.DiffusionReflect(particle_view[i], cell_view[celli, cellj, cellk])

        # particle[i].pos[0] += particle[i].vel[0]
        # particle[i].pos[1] += particle[i].vel[1]
        # particle[i].pos[2] += particle[i].vel[2]
        # if indice_1[i] == True:
        #     if particle[i].pos[0] >= cellSizeXYZ[0]:
        #         particle[i].pos[0] -= cellSizeXYZ[0]
        #     elif particle[i].pos[0] < 0:
        #         particle[i].pos[0] += cellSizeXYZ[0]

        #     if particle[i].pos[1] >= cellSizeXYZ[1]:
        #         particle[i].pos[1] -= cellSizeXYZ[1]
        #     elif particle[i].pos[1] < 0:
        #         particle[i].pos[1] += cellSizeXYZ[1]
        #     if (particle[i].pos[0] > cellSizeXYZ[0] or particle[i].pos[0] < 0 or
        #         particle[i].pos[1] > cellSizeXYZ[1] or particle[i].pos[1] < 0 or
        #         particle[i].pos[2] > cellSizeXYZ[2] or particle[i].pos[2] < 0):
        #         indice_1[i] = False
        # particle[i], indice_1[i] = boundary(particle[i], indice_1[i], cellSizeXYZ)

    # particle = particle[indice_1]





# def reaction_rate_parallel_all(
#     int[:,:,:,:] filmMatrix, 
#     int elements,
#     Particle[:] particles, 
#     Cell[:,:,:] cell, 
#     int[:] cellSizeXYZ):

#     cdef int particle_count = particles.shape[0]
#     cdef int i, celli, cellj, cellk
#     cdef double dot_product, angle_rad
#     cdef Particle[:] particle_view = particles
#     cdef Cell[:,:,:] cell_view = cell
#     cdef int[:,:,:] film_matrix_view = filmMatrix
#     cdef int[:] cell_size_xyz_view = cellSizeXYZ

#     # Create numpy arrays for intermediate results
#     cdef cnp.ndarray[cnp.int_t, ndim=3] update_film_etch = np.zeros(
#         (cell_size_xyz_view[0], cell_size_xyz_view[1], cell_size_xyz_view[2]), dtype=np.bool_
#     )
#     cdef cnp.ndarray[cnp.int_t, ndim=3] update_film_depo = np.zeros(
#         (cell_size_xyz_view[0], cell_size_xyz_view[1], cell_size_xyz_view[2]), dtype=np.bool_
#     )
#     cdef cnp.ndarray[cnp.int_t, ndim=1] reactList = np.full(particle_count, -1, dtype=np.int_)
#     cdef bint[:] indice_1 = np.ones(particle_count, dtype=np.bool_)
#     cdef cnp.ndarray[cnp.double_t, ndim=1] depo_particle = np.zeros(particle_count, dtype=np.double)

#     # Loop over all particles
#     for i in prange(particle_count, nogil=True):
#         celli = <int> particle_view[i].pos[0]
#         cellj = <int> particle_view[i].pos[1]
#         cellk = <int> particle_view[i].pos[2]

#         if cell_view[celli, cellj, cellk].id == 1:
#             # Retrieve the film at the current cell
#             # cdef int[:] film = film_matrix_view[celli, cellj, cellk]

#             # Compute dot product between particle velocity and cell normal
#             dot_product = (
#                 particle_view[i].vel[0] * cell_view[celli, cellj, cellk].normal[0] +
#                 particle_view[i].vel[1] * cell_view[celli, cellj, cellk].normal[1] +
#                 particle_view[i].vel[2] * cell_view[celli, cellj, cellk].normal[2]
#             )
#             dot_product = fabs(dot_product)
#             angle_rad = acos(dot_product)

#             # Calculate sticking probability
#             sticking_acceptList, particle_view[i] = absorb.sticking_probability_structed(
#                 particle_view[i], film_matrix_view[celli, cellj, cellk], angle_rad
#             )

        #     react_choice_indices = np.where(sticking_acceptList)[0]
        #     if react_choice_indices.size > 0:
        #         # Randomly select a reaction type
        #         react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
        #         reactList[i] = react_choice
        #         indice_1[i] = False

        #         # Determine reaction type and update particle or film
        #         react_type = absorb.react_type_table[particle_view[i].id, react_choice]

        #         if react_type == 1:  # Chemical transfer
        #             depo_particle[i] = 1
        #         elif react_type == 2:  # Physical sputtering
        #             depo_particle[i] = 2
        #         elif react_type == 3:  # Redeposition
        #             depo_particle[i] = 3
        #         elif react_type == 4:  # Thermal etching
        #             depo_particle[i] = 4
        #         elif react_type == 0:  # No reaction
        #             depo_particle[i] = 0

        #         # Update film based on reaction type
        #         if particle_view[i].id <= 1:  # Gas Cl Ion
        #             react_add = absorb.react_table_equation[particle_view[i].id, react_choice, :]
        #         else:  # Redeposited solid
        #             react_add = np.zeros(elements, dtype=np.int32)
        #             particle_index = particle_view[i].id - 2
        #             react_add[int(particle_index)] = 1

        #         if depo_particle[i] == 1 or depo_particle[i] == 4:  # Chemical transfer or remove
        #             film_matrix_view[celli, cellj, cellk] += react_add
        #             if np.sum(film_matrix_view[celli, cellj, cellk]) == 0:
        #                 update_film_etch[celli, cellj, cellk] = True
        #         elif depo_particle[i] == 2:  # Physical sputtering
        #             react_yield = sputter_yield.sputter_yield(
        #                 absorb.react_yield_p0[0], angle_rad, particle_view[i].E, 10
        #             )
        #             if react_yield > np.random.rand():
        #                 film_matrix_view[celli, cellj, cellk] += react_add
        #                 if np.sum(film_matrix_view[celli, cellj, cellk]) == 0:
        #                     update_film_etch[celli, cellj, cellk] = True

        #     # Save film back to memory
        #     # film_matrix_view[celli, cellj, cellk] = film
            
        #     # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
        #     if reactList[i] == -1:
        #         # particle[i, 3:6] = reflect.SpecularReflect(particle[i, 3:6], get_theta[i])
        #         particle[i].vel = reflection.DiffusionReflect(particle_view[i], cell_view[celli, cellj, cellk])

        # particle[i].pos[0] += particle[i].vel[0]
        # particle[i].pos[1] += particle[i].vel[1]
        # particle[i].pos[2] += particle[i].vel[2]
        # if indice_1[i] == True:
        #     if particle[i].pos[0] >= cellSizeXYZ[0]:
        #         particle[i].pos[0] -= cellSizeXYZ[0]
        #     elif particle[i].pos[0] < 0:
        #         particle[i].pos[0] += cellSizeXYZ[0]

        #     if particle[i].pos[1] >= cellSizeXYZ[1]:
        #         particle[i].pos[1] -= cellSizeXYZ[1]
        #     elif particle[i].pos[1] < 0:
        #         particle[i].pos[1] += cellSizeXYZ[1]
        #     if (particle[i].pos[0] > cellSizeXYZ[0] or particle[i].pos[0] < 0 or
        #         particle[i].pos[1] > cellSizeXYZ[1] or particle[i].pos[1] < 0 or
        #         particle[i].pos[2] > cellSizeXYZ[2] or particle[i].pos[2] < 0):
        #         indice_1[i] = False
        # particle[i], indice_1[i] = boundary(particle[i], indice_1[i], cellSizeXYZ)

    # particle = particle[indice_1]

    # return filmMatrix, particle, update_film_etch, update_film_depo, depo_particle



# # # reaction_rate_parallel_all 函数
# def reaction_rate_parallel_all( int [:,:,:,:] filmMatrix, 
#                                 int elements,
#                                 Particle [:] particle, 
#                                 Cell [:,:,:] cell, 
#                                 int [:] cellSizeXYZ):

#     cdef int particle_count = particle.shape[0]
#     particle_view: Particle[:] = particle
#     i: cython.Py_ssize_t
#     celli: cython.int
#     cellj: cython.int
#     cellk: cython.int
#     get_theta: double[3]
#     film: int[elements]
#     react_add: int[elements]
#     update_film_etch = np.zeros((cellSizeXYZ[0], cellSizeXYZ[1], cellSizeXYZ[2]), dtype=np.bool_)
#     update_film_depo = np.zeros((cellSizeXYZ[0], cellSizeXYZ[1], cellSizeXYZ[2]), dtype=np.bool_)
#     reactList = np.ones(particle.shape[0], dtype=np.int_) * -1
#     indice_1 = np.ones(particle.shape[0], dtype=np.bool_)
#     depo_particle = np.zeros(particle.shape[0])

#     for i in prange(particle_count):
#         celli = particle_view[i].px.astype(cython.int)
#         cellj = particle_view[i].py.astype(cython.int)
#         cellk = particle_view[i].pz.astype(cython.int)
#         if cell[celli, cellj, cellk].id == 1:
#             film = filmMatrix[celli, cellj, cellk]
#             dot_product = vel_normal_dot(particle[i], cell[celli, cellj, cellk])
#             dot_product = np.abs(dot_product)
#             angle_rad = np.arccos(dot_product)

#             sticking_acceptList, particle = sticking_probability_structed(particle[i], film, angle_rad)

#             react_choice_indices = np.where(sticking_acceptList)[0]
#             if react_choice_indices.size > 0:
#                 react_choice = react_choice_indices[np.random.randint(react_choice_indices.size)]
#                 reactList[i] = react_choice
#                 indice_1[i] = False
#                 react_type = react_type_table[particle, react_choice]

#                 if react_type == 1: # chemical transfer || p0 reaction type
#                     depo_particle[i] = 1
#                 elif react_type == 2: # physics sputter || p0 (E**2 - Eth**2) f(theta)
#                     depo_particle[i] = 2
#                 elif react_type == 3: # redepo
#                     depo_particle[i] = 3
#                 elif react_type == 4: # Themal etch || p0 reaction type
#                     depo_particle[i] = 4
#                 elif react_type == 0: # no reaction
#                     depo_particle[i] = 0

#             if int(particle[i, -1]) <= 1: # gas Cl Ion
#                 react_add = react_table_equation[int(particle[i, -1]), react_choice, :]
#             else: # redepo solid
#                 react_add = np.zeros(elements, dtype=np.int32)
#                 particle_index = int(particle[i, -1])-2
#                 react_add[int(particle_index)] = 1

#             if depo_particle[i] == 1: # chemical transfer
#                 film += react_add
#             if depo_particle[i] == 4: # chemical remove
#                 film += react_add
#                 if np.sum(film) == 0:
#                     # if film[i, 3] == 0:
#                     update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
#             if depo_particle[i] == 2: # physics sputter
#                 react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad, particle[i,-2], 10) # physics sputter || p0 (E**2 - Eth**2) f(theta)
#                 # react_yield = sputterYield.sputter_yield(react_yield_p0[0], angle_rad[i], particle[i,-2], film_Eth[int(reactList[i])])
#                 if react_yield > np.random.rand():
#                     film += react_add
#                     if np.sum(film) == 0:
#                         update_film_etch[ijk[0], ijk[1], ijk[2]] =  True
        
#             # if depo_particle[i] == 3: # depo
#             #     film_add_all = np.sum(react_add + film[i, :])
#             #     if film_add_all > film_density:
#             #         film_vaccum[i, :] += react_add
#             #         update_film_etch[int(particle[i, 6]), int(particle[i, 7]), int(particle[i, 8])] = True  
#             #     else:
#             #         film[i, :] += react_add
#             #         if np.sum(film[i, :]) == film_density:
#             #             update_film_depo[int(particle[i, 6]), int(particle[i, 7]), int(particle[i, 8])] = True
#             filmMatrix[ijk[0], ijk[1], ijk[2]] = film
#             # filmMatrix[get_plane_vaccum[i,0], get_plane_vaccum[i,1], get_plane_vaccum[i,2]] = film_vaccum
#             if reactList[i] == -1:
#                 # particle[i, 3:6] = reflect.SpecularReflect(particle[i, 3:6], get_theta[i])
#                 particle[i, 3:6] = reflect.DiffusionReflect(particle[i, 3:6], get_theta)

#         particle[i, :3] += particle[i, 3:6]
#         if indice_1[i] == True:
#             if particle[i, 0] >= cellSizeXYZ[0]:
#                 particle[i, 0] -= cellSizeXYZ[0]
#             elif particle[i, 0] < 0:
#                 particle[i, 0] += cellSizeXYZ[0]
#             if particle[i, 1] >= cellSizeXYZ[1]:
#                 particle[i, 1] -= cellSizeXYZ[1]
#             elif particle[i, 1] < 0:
#                 particle[i, 1] += cellSizeXYZ[1]
#             if (particle[i,0] > cellSizeXYZ[0] or particle[i,0] < 0 or
#                 particle[i,1] > cellSizeXYZ[1] or particle[i,1] < 0 or
#                 particle[i,2] > cellSizeXYZ[2] or particle[i,2] < 0):
#                 indice_1[i] = False
#         # particle[i], indice_1[i] = boundary(particle[i], indice_1[i], cellSizeXYZ)

#     particle = particle[indice_1]

#     return filmMatrix, particle, update_film_etch, update_film_depo, depo_particle


