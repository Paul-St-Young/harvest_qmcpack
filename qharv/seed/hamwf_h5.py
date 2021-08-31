# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to rw AFQMC hamiltonian and wavefunction hdf files
#  large matrices handled by scipy.sparse.csr_matrix (Compressed Sparse Row)
#  complex numbers handled by extra trailing dimension in hdf array

import numpy as np
from scipy.sparse import csr_matrix as csrm

# ======================== level 0: basic io ========================
from qharv.seed.wf_h5 import read, write, ls

def complex_view(float_array):
  shape = float_array.shape
  if shape[-1] != 2:
    msg = 'cannot handle data shape %s' % shape
    raise RuntimeError(msg)
  data = float_array.view(np.complex128).squeeze()
  return data

def float_view(complex_array):
  shape = complex_array.shape
  if not np.iscomplexobj(complex_array):
    msg = 'cannot handle array type %s' % complex_array.dtype
    raise RuntimeError(msg)
  data = complex_array.view(np.float64).reshape(shape+(2,))
  return data

def read_csrm(grp):
  """ Read CSR matrix from hdf group

  Args:
    grp (Group): h5py hdf group
  Return:
    csrm: sparse matrix
  Example:
    >>> fp = h5py.File('mat.h5', 'r')
    >>> g = fp['sparse_matrix']
    >>> mat = read_csrm(g)
  """
  # data could be real or complex
  darr = grp['data_'][()]
  dshape = darr.shape
  if len(dshape) == 1:
    data = darr
  elif (len(dshape) == 2):
    data = complex_view(darr)
  else:
    msg = 'read_csrm cannot handle data shape %s' % dshape
    raise RuntimeError(msg)
  # matrix indices
  indices = grp['jdata_'][()]
  lastptr = [grp['pointers_end_'][-1]]
  indptr = np.concatenate([grp['pointers_begin_'][()], lastptr])
  shape = grp['dims'][:2]
  mat = csrm((data, indices, indptr), shape=shape)
  return mat

def write_csrm(grp, mat):
  """ Write CSR matrix to hdf group

  Args:
    grp (Group): h5py hdf group
    mat (csrm): sparse matrix
  Example:
    >>> fp = h5py.File('mat.h5', 'w')
    >>> g = fp.create_group('sparse_matrix')
    >>> write_csrm(g, mat)
  """
  dims = mat.shape + (mat.nnz,)
  grp['data_'] = float_view(mat.data)
  grp['jdata_'] = mat.indices
  grp['pointers_begin_'] = mat.indptr[:-1]
  grp['pointers_end_'] = mat.indptr[1:]

# =========================== level 1: FFT ==========================
def cubic_pos(spaces):
  ndim = len(spaces)
  gvecs = np.stack(
    np.meshgrid(*spaces, indexing='ij'), axis=-1
  ).reshape(-1, ndim)
  return gvecs

def get_rvecs(axes, mesh):
  spaces = [np.arange(nx) for nx in mesh]
  gvecs = cubic_pos(spaces)
  fracs = axes/mesh
  return np.dot(gvecs, fracs)

def get_kvecs(raxes, mesh):
  spaces = [np.fft.fftfreq(nx)*nx for nx in mesh]
  gvecs = cubic_pos(spaces)
  return np.dot(gvecs, raxes)

def calc_eikr(kvecs, rvecs):
  kdotr = np.einsum('...i,ri->...r', kvecs, rvecs)
  eikr = np.exp(1j*kdotr)
  return eikr
