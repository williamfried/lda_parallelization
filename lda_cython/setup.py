from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

np_path = np.get_include()
ext_modules = [
    Extension(
        "sampler", ["sampler.pyx"],
    ),
    Extension(
        "cgs", ["cgs.pyx"],
        extra_compile_args=[
            "-fopenmp",
            # Note these args are for MPI, unknown if specific to system
            "-Wl,-flat_namespace", "-Wl,-commons,use_dylibs"
        ], extra_link_args=[
            "-fopenmp",
            # Note these args are for MPI, likely specific to system
            # as well as implementation of MPI
            "-I/usr/local/Cellar/mpich/3.3.2/include",
            "-L/usr/local/Cellar/mpich/3.3.2/lib", "-lmpi", "-lpmpi"
        ],
    )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np_path, ],
)

