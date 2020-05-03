from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

np_path = np.get_include()
ext_modules = [
    Extension(
        "utils", ["utils.pyx"],
    ),
    Extension(
        "cgs", ["cgs.pyx"],
        extra_compile_args=[
            # "-fopenmp",
            # Note these args are for MPI, unknown if specific to system
            # To replace manually,  see
            # https://www.open-mpi.org/doc/v3.0/man1/mpicc.1.php
            # or https://www.mpich.org/static/docs/v3.1.x/www1/mpicc.html
            # and show args
            "-Wl,-flat_namespace", "-Wl,-commons,use_dylibs"
        ], extra_link_args=[
            # "-fopenmp",
            # Note these args are for MPI, likely specific to system
            # as well as implementation of MPI
            # To replace these manually, see note above
            "-I/usr/local/Cellar/mpich/3.3.2/include",
            "-L/usr/local/Cellar/mpich/3.3.2/lib", "-lmpi", "-lpmpi"
        ],
    )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np_path, ],
)

