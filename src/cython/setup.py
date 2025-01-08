# from setuptools import setup
# from Cython.Build import cythonize
# import numpy as np

# setup(
#     ext_modules=cythonize("plane_index_fast_cython.pyx"),
#     include_dirs=[np.get_include()]
# )

# from setuptools import setup
# from setuptools.extension import Extension
# from Cython.Build import cythonize
# import numpy as np

# extensions = [
#     Extension(
#         "get_plane_vaccum",
#         ["get_plane_vaccum.pyx"],
#         include_dirs=[np.get_include()],
#         extra_compile_args=["/openmp"],  # 对于 Windows
#         extra_link_args=["/openmp"],     # 对于 Windows
#     )
# ]

# setup(
#     ext_modules=cythonize(extensions, language_level=3),
# )

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("example.pyx"),
    include_dirs=[np.get_include()]
)