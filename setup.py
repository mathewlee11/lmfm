from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

setup(author='mathewlee11',
      version='0.4.1',
      author_email='mathewlee11@gmail.com',
      name='lmfm',
      packages=['lmfm'],
      url='https://github.com/mathewlee11/lmfm',
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("lmfm_fast", ["lmfm/lmfm_fast.pyx"],
                   libraries=[], include_dirs=[numpy.get_include()])])
