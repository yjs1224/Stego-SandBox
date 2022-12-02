# cython: language_level=3
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
# import cython_utils

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules = cythonize(["cython_sampler.pyx","cython_utils.pyx","norm_aggr.pyx"]), include_dirs = [numpy.get_include()])
# to compile: python graphsaint/setup.py build_ext --inplace
