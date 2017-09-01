#from distutils.core import setup
from setuptools import setup, find_packages

setup(
  name           = 'harvest_qmcpack',
  version        = '0.0',
  description    = 'Routines to inspect and modify QMCPACK objects.',
  author         = 'Yubo "Paul" Yang',
  author_email   = 'yyang173@illinois.edu',
  url            = 'http://publish.illinois.edu/yubo-paul-yang/',
  package_dir    = {'qharv':'qharv'},
  packages       = find_packages(),
  install_requires = []
)
