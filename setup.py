import numpy
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext


# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(name='lmfm',
      version='0.1',
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize("lmfm.pyx"),
      include_dirs=[numpy.get_include()],
      author='mathewlee11',
      author_email='mathewlee11@gmail.com')
