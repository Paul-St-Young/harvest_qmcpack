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
  dims = (mat.shape[1], mat.shape[0], mat.nnz)
  grp['data_'] = float_view(mat.data)
  grp['jdata_'] = mat.indices
  grp['pointers_begin_'] = mat.indptr[:-1]
  grp['pointers_end_'] = mat.indptr[1:]
  grp['dims'] = dims

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

# =========================== level 2: ERI ==========================

def calc_pair_densities_on_fftgrid(ukl, gvl, raxes, rvecs):
  """ Calculate pair densities given a list of Bloch functions "ukl".
  Each u(k) should be given by PW Miller indices "gv" and rec. latt. "raxes".
  Currently require user provided real-space FFT grid points.

  Example:
  >>> gvl = [np.array([[0, 0, 0], [0, 0, 1]])]*2
  >>> ukl = [[np.array([1+0j, 0]]), np.array([[0.5, 0.5+0.1j]])]
  >>> axes = alat*np.eye(3)
  >>> mesh = (15, 15, 15)
  >>> rvecs = get_rvecs(axes, mesh)
  >>> raxes = 2*np.pi*np.linalg.inv(axes).T
  >>> Pij = calc_pair_densities_on_fftgrid(ukl, gvl, raxes, rvecs)
  """
  nbndl = [len(uk) for uk in ukl]
  nbnd = nbndl[0]  # !!!! assume same nbnd at all kpts
  if not np.allclose(nbndl, nbnd):
    msg = 'pair densities for varying nbnd'
    raise NotImplementedError(msg)
  ngrid = len(rvecs)  # np.prod(mesh)
  nkpt = len(gvl)
  Pijs = np.zeros([nkpt, nkpt, nbnd, nbnd, ngrid], dtype=np.complex128)
  for ik in range(nkpt):
    kvi = np.dot(gvl[ik], raxes)
    for jk in range(ik, nkpt):
      kvj = np.dot(gvl[jk], raxes)
      use_symm = ik == jk
      # put pair density on real-space FFT grid
      Pij = calc_pij_on_fftgrid(rvecs, kvi, ukl[ik], kvj, ukl[jk],
                                use_symm=use_symm)
      Pijs[ik, jk] = Pij
      if not use_symm:  # off diagonal in kpts, copy to hermitian
        Pijs[jk, ik] = Pij.conj().transpose([1, 0, 2])
  return Pijs

def calc_pij_on_fftgrid(rvecs, kvecs0, uk0, kvecs1, uk1, use_symm=False):
  ngrid = len(rvecs)
  nstate0, npw0 = uk0.shape
  nstate1, npw1 = uk1.shape
  eikr0 = calc_eikr(kvecs0, rvecs)
  eikr1 = calc_eikr(kvecs1, rvecs)
  Pij = np.zeros([nstate0, nstate1, ngrid], dtype=np.complex128)
  for i in range(nstate0):
    uuic = np.dot(uk0[i], eikr0).conj()
    j0 = i if use_symm else 0
    for j in range(j0, nstate1):
      uuj = np.dot(uk1[j], eikr1)
      val = uuic*uuj
      Pij[i, j, :] = val
      if use_symm and (i != j):
        Pij[j, i, :] = val.conj()
  return Pij
