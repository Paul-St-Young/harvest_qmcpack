# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 walker data output. Mostly built around PyTables.
import os
import tables
import numpy as np


def extract_checkpoint_walkers(fconfig):
  """ extract the checkpoint walkers from config.h5 file

  Args:
    fconfig (str): path to config.h5 file
  Return:
    np.array: a list of walkers of shape (nconf, nptcl, ndim)
  """
  h5file = tables.open_file(fconfig)
  walkers = h5file.root.state_0.walkers.read()
  h5file.close()
  return walkers


def save_mat(mat, h5file, slab, name):
  """ save matrix as floating point numbers in h5file.slab.name

  for /name, use slab = h5file.root

  Args:
    mat (np.array): 2D numpy array of floats
    h5file (tables.File): get from tables.open_file(fname, 'w')
    slab (tables.Group): HDF5 slab, could be root slab
    name (str): name of CArray to create
  """
  atom = tables.Float64Atom()
  ca = h5file.create_carray(slab, name, atom, mat.shape)
  ca[:, :] = mat
