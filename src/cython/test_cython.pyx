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

# 定义一个线程独立的随机数生成器
cdef class ThreadLocalRNG:
    cdef mt19937 rng
    cdef uniform_real_distribution[double] dist

    def __cinit__(self):
        # 使用随机设备初始化
        from libcpp.random cimport random_device
        cdef random_device rd
        self.rng = mt19937(rd())
        self.dist = uniform_real_distribution[double](0.0, 1.0)

    cdef double random(self) noexcept nogil:
        # 返回一个均匀分布的随机数
        return self.dist(self.rng)

# 使用多线程生成随机数
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_random_numbers(size, threads=4):
    cdef int i, j
    cdef int chunk_size = size // threads
    cdef cnp.ndarray[cnp.double_t, ndim=1] results = np.zeros(size, dtype=np.double)
    cdef vector[ThreadLocalRNG] rngs

    # 每个线程初始化自己的随机数生成器
    rngs.resize(threads)
    with nogil, parallel():
        for i in prange(threads, nogil=True):
            rngs[i] = ThreadLocalRNG()

        for i in prange(size, nogil=True):
            # 获取当前线程 ID
            thread_id = i // chunk_size
            if thread_id >= threads:
                thread_id = threads - 1
            # 使用当前线程的随机数生成器
            results[i] = rngs[thread_id].random()

    return results



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