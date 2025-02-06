# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "react",
        ["react.cpp"],
        # ["Cell.cpp"],
        include_dirs=['/usr/include/eigen3'],  # Eigen路径
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],

    ),
]

setup(
    name="react",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)




# # Available at setup time due to pyproject.toml
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# from setuptools import setup


# ext_modules = [
#     Pybind11Extension(
#         "simulation",
#         ["simulation.cpp"],
#         # ["Cell.cpp"],
#         include_dirs=['/usr/include/eigen3'],  # Eigen路径
#         extra_compile_args=["-fopenmp"],
#         extra_link_args=["-fopenmp"],

#     ),
# ]

# setup(
#     name="simulation",
#     ext_modules=ext_modules,
#     extras_require={"test": "pytest"},
#     cmdclass={"build_ext": build_ext},
#     zip_safe=False,
#     python_requires=">=3.7",
# )