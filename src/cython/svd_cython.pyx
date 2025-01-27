# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False

import numpy as np
cimport numpy as cnp
from libcpp cimport bool  # 关键修改：引入C++的bool类型

# 声明C++接口
cdef extern from "eigen_svd.h":
    void eigen_compute_svd(
        const double* input,
        int rows,
        int cols,
        double* U,
        double* S,
        double* V,
        bool full_matrices  # 使用C++的bool类型
    ) nogil  # 添加异常传播

def compute_svd(cnp.ndarray[double, ndim=2] arr, full_matrices=False):
    # 确保内存布局连续
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    
    # 获取数组维度
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    
    # 类型转换必须在GIL保护下进行
    cdef bool c_full_matrices = full_matrices  # 关键修改：显式类型转换
    
    # 预分配输出缓冲区
    cdef int min_dim = min(rows, cols)
    cdef int U_cols = min_dim if not c_full_matrices else rows
    cdef int V_rows = min_dim if not c_full_matrices else cols
    
    cdef cnp.ndarray[double, ndim=2] U = np.empty((rows, U_cols), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] S = np.empty(min_dim, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] V = np.empty((V_rows, cols), dtype=np.float64)
    
    # 调用C++函数
    with nogil:
        eigen_compute_svd(
            &arr[0,0],
            rows,
            cols,
            &U[0,0],
            &S[0],
            &V[0,0],
            c_full_matrices  # 传递已转换的C++ bool
        )
    
    return U, S, V.T