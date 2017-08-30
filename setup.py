from distutils.core import setup

setup(
  name           = 'harvest_qmcpack',
  version        = '0.0',
  description    = 'Routines to inspect and modify QMCPACK objects.',
  author         = 'Yubo "Paul" Yang',
  author_email   = 'yyang173@illinois.edu',
  url            = 'http://publish.illinois.edu/yubo-paul-yang/',
  packages       = ['qharv','qharv.seed','qharv.cross','qharv.inspect','qharv.reel'],
  install_requires = []
)
