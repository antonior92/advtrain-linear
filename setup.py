from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name='advtrain-linear',
      version='0.1',
      description='Routines for ',
      author='Antonio H. Ribeiro',
      author_email='antonior92@gmail.com',
      packages=['linadvtrain'],
      ext_modules=cythonize("linadvtrain/first_order_methods.pyx"),
      include_dirs=[numpy.get_include()]
      )