# distutils: language = c++

from cython.parallel cimport parallel, prange
from libcpp.vector cimport vector
from libcpp.algorithm cimport nth_element, max_element, remove
cimport cython
from cython.operator cimport dereference
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np



from libcpp.random cimport mt19937, uniform_real_distribution
cimport numpy as cnp


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




@cython.boundscheck(False)
@cython.wraparound(False)
def median_along_axis0(const double[:,:] x):
    cdef double[::1] out = np.empty(x.shape[1])
    cdef Py_ssize_t i, j

    cdef vector[double] *scratch
    cdef vector[double].iterator median_it
    with nogil, parallel():
        # allocate scratch space per loop
        scratch = new vector[double](x.shape[0])
        try:
            for i in prange(x.shape[1]):
                # copy row into scratch space
                for j in range(x.shape[0]):
                    dereference(scratch)[j] = x[j, i]
                median_it = scratch.begin() + scratch.size()//2
                nth_element(scratch.begin(), median_it, scratch.end())
                # for the sake of a simple example, don't handle even lengths...
                out[i] = dereference(median_it)
        finally:
            del scratch
    return np.asarray(out)

@cython.boundscheck(False)
@cython.wraparound(False)
def test_pointer(const double[:,:] x):
    cdef double[::1] out = np.zeros(x.shape[1])
    cdef Py_ssize_t i, j

    cdef vector[double] *scratch
    cdef vector[int] *x0
    cdef vector[double].iterator median_it
    with nogil, parallel():
        # allocate scratch space per loop
        scratch = new vector[double](x.shape[0])
        x0 = new vector[int](1)
        try:
            for i in prange(x.shape[1]):
                dereference(x0)[0] = <int> x[0, i]
                # copy row into scratch space
                if dereference(x0)[0] == 0:
                    for j in range(x.shape[0]):
                        dereference(scratch)[j] = x[j, i]
                    median_it = scratch.begin() + scratch.size()//2
                    nth_element(scratch.begin(), median_it, scratch.end())
                    # for the sake of a simple example, don't handle even lengths...
                    out[i] = dereference(median_it)
                else:
                    continue
        finally:
            del scratch
            del x0
    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
def test_pointer_2(const double[:,:] x):
    cdef double[::1] out = np.zeros(x.shape[1])
    cdef Py_ssize_t i, j

    cdef vector[double] *scratch
    cdef int *x0
    cdef vector[double].iterator median_it
    with nogil, parallel():
        # allocate scratch space per loop
        scratch = new vector[double](x.shape[0])
        x0 = <int*>malloc(sizeof(int))
        try:
            for i in prange(x.shape[1]):
                # dereference(x0)[0] = <int> x[0, i]
                x0[0] = <int> x[0, i]
                # copy row into scratch space
                if x0[0] == 0:
                    for j in range(x.shape[0]):
                        dereference(scratch)[j] = x[j, i]
                    median_it = scratch.begin() + scratch.size()//2
                    nth_element(scratch.begin(), median_it, scratch.end())
                    # for the sake of a simple example, don't handle even lengths...
                    out[i] = dereference(median_it)
                else:
                    continue
        finally:
            del scratch
            free(x0)
    return np.asarray(out)






from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

cdef int* container(int a) noexcept nogil:
    """
    示例函数，动态分配一个大小为 5 的整型数组，返回指针
    """
    cdef int* arr = <int*>malloc(5 * sizeof(int))
    cdef int i
    for i in range(5):
        arr[i] = a  # 示例赋值：数组元素为 [0, 1, 2, 3, 4]
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
def test_pointer_with_prange(const double[:,:] x):
    """
    多线程中使用指针传递临时空间
    """
    cdef double[::1] out = np.zeros(x.shape[1])  # 输出数组
    cdef Py_ssize_t i, j
    cdef int *scratch  # 临时空间指针
    cdef int *x0

    with nogil, parallel():
        try:
            x0 = <int*>malloc(sizeof(int))  # 动态分配一个 int 空间
            # scratch = <int*>malloc(5*sizeof(int)) 
            for i in prange(x.shape[1]):
                x0[0] = <int> x[0, i]  # 读取数据到临时空间
                if x0[0] == 0:
                    # 调用函数动态分配数组并传递到临时空间
                    scratch = container(i)
                    # 示例操作：将第一个值存入输出
                    out[i] = scratch[2]

                    # 释放临时空间
                    free(scratch)
                else:
                    continue
        finally:
            free(x0)  # 确保释放动态分配的指针
            # free(scratch)
    return np.asarray(out)


def max_value_in_vector():
    cdef vector[int] my_vector
    cdef vector[int].iterator max_it

    # 初始化 vector
    my_vector.push_back(1)
    my_vector.push_back(5)
    my_vector.push_back(3)

    # 使用 max_element 找到最大值
    max_it = max_element(my_vector.begin(), my_vector.end())
    return dereference(max_it)  # 返回最大值



def vector_to_numpy():
    cdef vector[vector[int]] my_vector
    cdef vector[int] temp_vector
    cdef int rows, cols, i, j

    # 构造临时 vector 数据
    temp_vector = vector[int]()
    temp_vector.push_back(1)
    temp_vector.push_back(2)
    temp_vector.push_back(3)
    my_vector.push_back(temp_vector)

    temp_vector.clear()
    temp_vector.push_back(4)
    temp_vector.push_back(5)
    temp_vector.push_back(6)
    my_vector.push_back(temp_vector)

    temp_vector.clear()
    temp_vector.push_back(7)
    temp_vector.push_back(8)
    temp_vector.push_back(9)
    my_vector.push_back(temp_vector)

    # 转换为 NumPy 数组
    rows = my_vector.size()  # 获取行数
    cols = my_vector[0].size() if rows > 0 else 0  # 获取列数
    cdef cnp.ndarray[cnp.int32_t, ndim=2] np_array = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            np_array[i, j] = my_vector[i][j]

    return np_array


def remove_from_vector(int target):
    cdef vector[int] my_vector

    # 初始化 vector
    my_vector.push_back(1)
    my_vector.push_back(2)
    my_vector.push_back(3)
    my_vector.push_back(2)

    # 使用 remove 移除所有值等于 target 的元素
    my_vector.erase(remove(my_vector.begin(), my_vector.end(), target), my_vector.end())

    # 打印移除后的结果
    for value in my_vector:
        print(value)


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_to_numpy_parallel():
    cdef int rows = 100, cols = 3
    cdef int i, j
    cdef vector[int] *temp_vector  # 每个线程独立的临时 vector
    # 创建用于存储结果的主线程 vector
    cdef vector[vector[int]] my_vector
    my_vector.resize(rows)  # 提前分配空间

    # 多线程填充数据
    with nogil, parallel():
        temp_vector = new vector[int](3)
        try:
            for i in prange(rows):
                
                dereference(temp_vector)[0] = i
                dereference(temp_vector)[1] = i+1 
                dereference(temp_vector)[2] = i+2
                my_vector[i] = dereference(temp_vector)  # 将临时 vector 存储到主 vector 中
        finally:
            del temp_vector

    # 转换为 NumPy 数组
    cdef cnp.ndarray[cnp.int32_t, ndim=2] np_array = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            np_array[i, j] = my_vector[i][j]

    return np_array

@cython.boundscheck(False)
@cython.wraparound(False)
def vector_to_numpy_parallel_safe():
    cdef int rows = 100, cols = 3
    cdef int i, j
    cdef vector[vector[int]] my_vector
    cdef vector[int] *temp_vector  # 每个线程独立的临时 vector
    # 多线程填充数据
    with nogil, parallel():
        temp_vector = new vector[int](3)
        try:
            for i in prange(rows):

                dereference(temp_vector)[0] = i
                dereference(temp_vector)[1] = i+1 
                dereference(temp_vector)[2] = i+2

                # 使用 GIL 将临时数据添加到主线程 vector
                with gil:
                    my_vector.push_back(dereference(temp_vector))
        finally:
            del temp_vector
    # 转换为 NumPy 数组
    cdef cnp.ndarray[cnp.int32_t, ndim=2] np_array = np.zeros((len(my_vector), cols), dtype=np.int32)
    for i in range(len(my_vector)):
        for j in range(len(my_vector[i])):
            np_array[i, j] = my_vector[i][j]

    return np_array


