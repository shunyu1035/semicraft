import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "reflection",
        ["reflection.pyx"],
        include_dirs=[np.get_include(),
                      "D:/ysy/SimProfile/src/cython/header"],
        annotate=True,  # 生成 HTML 注释文件，便于调试和性能分析
        extra_compile_args=[
        "/openmp",
        "/O2",            # 开启完整优化
        "/arch:AVX2",     # 启用 AVX2 指令集
        "/fp:fast",       # 浮点运算优化（可能牺牲精度）
        "/GL",            # 启用全局优化
        "/D_USE_MATH_DEFINES"  # 允许数学定义（如 M_PI）
    ],
        extra_link_args=["/openmp"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)