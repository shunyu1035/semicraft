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
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(ext_modules, gdb_debug=True),
)








