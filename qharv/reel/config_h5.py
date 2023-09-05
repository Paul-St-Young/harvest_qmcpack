# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 walker data output. Mostly built around PyTables.
import os
import tables
import numpy as np

def open_read(fname, mode='r'):
  fp = tables.open_file(fname, mode=mode)
  return fp

def open_write(fname):
  filters = tables.Filters(complevel=5, complib='zlib')
  fp = tables.open_file(fname, mode='w', filters=filters)
  return fp

def read_group(grp):
  if type(grp) is tables.group.Group:
    data = dict()
    for g1 in grp:
      data[g1._v_name] = read_group(g1)
    return data
  else:
    return grp.read()

def load_dict(fname):
  data = dict()
  with open_read(fname) as h5file:
    for grp in h5file.root:
      data[grp._v_name] = read_group(grp)
    h5file.close()
  return data

def write_dict(fname, data):
  with open_write(fname) as h5file:
    save_dict(data, h5file)

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
  fp = open_write(fname)
  save_vec(mat, fp, fp.root, name)
  fp.close()

def loadh5(fname, path='/data'):
  """ load matrix from h5 file, mimic np.loadtxt

  Args:
    fname (str): name of hdf5 to read
  Return:
    np.array: matrix of data
  """
  fp = open_read(fname)
  slab = fp.get_node(path)
  mat = slab.read()
  fp.close()
  return mat

def save_vec(vec, h5file, slab, name):
  """ save numpy array into an h5 slab under name

  Args:
    vec (np.array): numpy ndarray of arbitrary dimension and type
    h5file (tables.file.File): pytables File
    slab (tables.Group): HDF5 slab
    name (str): name of CArray to create
  """
  try:
    len(vec)
  except TypeError as err:
    vec = np.array([vec])
  atom = tables.Atom.from_dtype(vec.dtype)
  ca = h5file.create_carray(slab, name, atom, vec.shape)
  ca[:] = vec

def save_dict(arr_dict, h5file, slab=None):
  """ save a dictionary of numpy arrays into h5file
   each entry will create its own sub-slab using key as name

  Args:
    arr_dict (dict): dictionary of numpy arrays
    h5file (tables.file.File): pytables File
    slab (tables.Group, optional): HDF5 slab, if None, then use root
  """
  if slab is None:
    slab = h5file.root
  for key, arr in arr_dict.items():
    if type(arr) is dict:
      slab1 = h5file.create_group(slab, key)
      save_dict(arr, h5file, slab=slab1)
    else:
      save_vec(arr, h5file, slab, key)
  h5file.flush()

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
