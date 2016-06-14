from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(author='mathewlee11',
      version='0.5.0',
      author_email='mathewlee11@gmail.com',
      name='lmfm',
      packages=['lmfm'],
      url='https://github.com/mathewlee11/lmfm',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("als_fast", ["lmfm/als_fast.pyx"],
                   libraries=[], include_dirs=[numpy.get_include()]),
                   Extension("sgd_fast", ["lmfm/sgd_fast.pyx"],
                   libraries=[], include_dirs=[numpy.get_include()])])
