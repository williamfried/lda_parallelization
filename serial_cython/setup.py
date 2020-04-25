from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

np_path = np.get_include()
ext_modules = [
    Extension(
        "sampler", ["sampler.pyx"],
    ),
    Extension(
        "serial_cgs_cython", ["serial_cgs_cython.pyx"],
    )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np_path, ],
)
