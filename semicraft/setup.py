# Cross-platform setup: set MSVC to use UTF-8 and platform-appropriate flags
import os
import sys
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext as _build_ext


class build_ext(_build_ext):
    """Customize build to set compiler-specific flags at build time."""

    def build_extensions(self):
        compiler_type = getattr(self.compiler, 'compiler_type', None)
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or [])
            ext.extra_link_args = list(ext.extra_link_args or [])

            if compiler_type == 'msvc':
                # Ensure MSVC treats source as UTF-8 (fixes C4819-related parse errors)
                if '/utf-8' not in ext.extra_compile_args:
                    ext.extra_compile_args.append('/utf-8')
                # Optimization and OpenMP for MSVC
                if '/O2' not in ext.extra_compile_args:
                    ext.extra_compile_args.append('/O2')
                # define math constants like M_PI
                if '/D_USE_MATH_DEFINES' not in ext.extra_compile_args:
                    ext.extra_compile_args.append('/D_USE_MATH_DEFINES')
                # enable OpenMP unless user disables it
                if os.environ.get('DISABLE_OPENMP', '0') not in ('1', 'true', 'True'):
                    if '/openmp' not in ext.extra_compile_args:
                        ext.extra_compile_args.append('/openmp')

            else:
                # GCC/Clang/MinGW on unix-like platforms
                if '-O3' not in ext.extra_compile_args:
                    ext.extra_compile_args += ['-O3', '-g']
                if os.environ.get('DISABLE_OPENMP', '0') not in ('1', 'true', 'True'):
                    # macOS default clang may not have OpenMP; user can install libomp if needed
                    if sys.platform != 'darwin':
                        if '-fopenmp' not in ext.extra_compile_args:
                            ext.extra_compile_args.append('-fopenmp')
                            ext.extra_link_args.append('-fopenmp')

        super().build_extensions()


eigen_dir = os.environ.get('EIGEN3_INCLUDE_DIR', r"E:\deps\eigen-3.4.0")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))

ext_modules = [
    Pybind11Extension(
        "semicraft",
        sorted(glob("src/*.cpp")),  # all source files
        include_dirs=[eigen_dir, src_dir],
    ),
]


setup(
    name="semicraft",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

