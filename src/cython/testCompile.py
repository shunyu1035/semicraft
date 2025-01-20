import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "test_cython",
        ["test_cython.pyx"],
        include_dirs=[np.get_include()],
        language="c++",  # 使用 C++ 编译
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)