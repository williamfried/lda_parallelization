from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

np_path = np.get_include()
ext_modules = [
    Extension(
        "sampler", ["sampler.pyx"],
        # extra_compile_args=['-fopenmp'],extra_link_args=['-fopenmp']
    )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np_path,]
)

