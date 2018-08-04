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
  """ save matrix in h5file.slab.name

  for /name, use slab = h5file.root
  see example usage in saveh5

  Args:
    mat (np.array): 2D numpy array of floats
    h5file (tables.File): get from tables.open_file(fname, 'w')
    slab (tables.Group): HDF5 slab, could be root slab
    name (str): name of CArray to create
  """
  atom = tables.Atom.from_dtype(mat.dtype)
  ca = h5file.create_carray(slab, name, atom, mat.shape)
  ca[:, :] = mat


def save_vec(vec, h5file, slab, name):
  atom = tables.Atom.from_dtype(vec.dtype)
  ca = h5file.create_carray(slab, name, atom, vec.shape)
  ca[:] = vec


def saveh5(fname, mat, name='data'):
  """ save matrix at root of h5 file, mimic call signature of np.savetxt

  e.g. mat = np.eye(3)
  saveh5('mat.h5', mat)
  $ h5ls mat.h5
  data               Dataset {3/2730, 3}

  Args:
    fname (str): name of hdf5 file to write
    mat (np.array): 2D numpy array of floats
    name (str, optional): CArray name at the root of the hdf5 file
  """
  filters = tables.Filters(complevel=5, complib='zlib')
  fp = tables.open_file(fname, mode='w', filters=filters)
  save_mat(mat, fp, fp.root, name)
  fp.close()


def loadh5(fname, path='/data'):
  """ load matrix from h5 file, mimic np.loadtxt

  Args:
    fname (str): name of hdf5 to read
  Return:
    np.array: matrix of data
  """
  fp = tables.open_file(fname, mode='r')
  slab = fp.get_node(path)
  mat = slab.read()
  fp.close()
  return mat
