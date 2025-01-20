import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "absorb",
        ["absorb.pyx"],
        include_dirs=[np.get_include(),
                      "D:/ysy/SimProfile/src/cython/header"],
        extra_compile_args=["/openmp"],
        extra_link_args=["/openmp"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)