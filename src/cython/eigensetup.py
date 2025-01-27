from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "svd_cython",
    sources=["svd_cython.pyx", "eigen_svd.cpp"],  # 包含C++源文件
    include_dirs=[np.get_include(),'/usr/include/eigen3'],  # Eigen路径
    language="c++",
    extra_compile_args=["-std=c++17", "-O3", "-march=native"],
    extra_link_args=["-std=c++17"]
)

setup(
    name="svd_cython",
    ext_modules=cythonize(ext, compiler_directives={'language_level': "3"}),
)