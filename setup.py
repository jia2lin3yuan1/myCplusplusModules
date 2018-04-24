#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension


from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

baseDir = "/home/yuanjial/Projects/Python-pure/instanceinference/code/Cython/cython_segments/"
src_files=["py_distPropsGenerate.pyx", baseDir+"include/svm/svm.cpp"]
src_include = [np.get_include(), baseDir+"include/"]
src_libs = []
src_link_args = ['-g']
#src_compile_args = ["-std=c++11", "-DIL_STD", "-Wall", "-ggdb"]
src_compile_args=['-Wno-sign-compare', '-DILOUSESTL','-DIL_STD','-std=c++1y','-O3','-DHAVE_CPP11_INITIALIZER_LISTS']


ext1 = Extension( "py_distPropsGenerate",
                  src_files,
                  language="c++",
                  include_dirs=src_include,
                  libraries=src_libs,
                  extra_compile_args=src_compile_args,
                  extra_link_args=src_link_args )

setup(cmdclass={'build_ext':build_ext}, ext_modules=cythonize([ext1]))
#setup(name='Py_distPropsGenerate', ext_modules=cythonize([ext1]))
