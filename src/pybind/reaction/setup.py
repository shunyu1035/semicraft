# from setuptools import setup, Extension
# import pybind11

# # 定义 C++ 扩展模块
# ext_modules = [
#     Extension(
#         name='film_optimized',  # Python 模块名
#         sources=['film_optimized.cpp'],  # C++ 源码文件
#         include_dirs=[
#             pybind11.get_include(),  # PyBind11 头文件路径
#             ['/usr/include/eigen3']                  # Eigen 头文件路径
#         ],
#         # include_dirs=['/usr/include/eigen3'],  # Eigen路径
#         language='c++',
#         extra_compile_args=['-std=c++17', '-O3', '-fopenmp'],  # 编译优化选项
#         extra_link_args=['-fopenmp']  # 链接 OpenMP
#     )
# ]

# # 构建配置
# setup(
#     name='film_optimized',
#     version='1.0',
#     ext_modules=ext_modules,
#     install_requires=['pybind11>=2.6'],  # 依赖声明
#     python_requires='>=3.6'
# )


# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# __version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "react",
        ["react.cpp"],
        include_dirs=['/usr/include/eigen3'],  # Eigen路径
        # Example: passing in the version to the compiled code
        # define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="react",
    # version=__version__,
    # long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
