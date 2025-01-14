import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "test_cython",
        ["test_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/openmp"],
        extra_link_args=["/openmp"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)