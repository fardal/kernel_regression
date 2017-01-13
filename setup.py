#!/usr/bin/env python

from distutils.core import setup

raise Exception('setup.py needs work.')

setup(name='kernel_regression',
      version='1.0',
      description='Implementation of locally constant and linear kernel regression with automatic bandwidth selection compatible with sklearn.',
      author='Mark Fardal'
      author_email='fardal@stsci.edu',
      url='https://github.com/fardal/kernel_regression',
      py_modules = ['kernel_regression']
     )
