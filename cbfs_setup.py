#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
print("hey")
print(build_ext)
print("hey")
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "cbfs", sources=["cbfs.pyx"], include_dirs=[numpy.get_include()])
    ],)
