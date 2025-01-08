# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp

# 被调用的函数
def func_b(cnp.ndarray[cnp.float64_t, ndim=1] array):
    cdef int i
    cdef double result = 0.0
    for i in range(array.shape[0]):
        result += array[i]
    return result

# 调用 func_b 的函数
def func_a(cnp.ndarray[cnp.float64_t, ndim=1] array):
    cdef double sum_result = func_b(array)
    print(f"The sum is: {sum_result}")
    return sum_result
