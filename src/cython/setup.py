import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'reaction_parallel_cython',
        ['reaction_parallel_cython.pyx'],
        include_dirs=[np.get_include()],
                    #   "D:/ysy/SimProfile/src/cython/header"],
                    #   "./header"],
        language="c++",  # 使用 C++ 编译
        # extra_compile_args=["/openmp"],
        # extra_link_args=["/openmp"],
        extra_compile_args=[
            "-std=c++17",         # 使用 C++17 标准
            "-O3",                # 开启最高级别优化
            "-ffast-math",        # 启用快速数学优化（可能会降低精度）
            "-march=native",      # 使用目标机器的所有优化特性
            "-mtune=native",      # 优化特定 CPU 的指令集
            "-mavx2",             # 启用 AVX2 SIMD 指令集
            "-mfma",              # 启用 FMA（浮点融合乘加）
            "-funroll-loops",     # 循环展开
            "-fopenmp",           # 启用 OpenMP（多线程并行）
        ],
        # extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True),
)








