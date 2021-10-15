# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to rw AFQMC hamiltonian and wavefunction hdf files
#  large matrices handled by scipy.sparse.csr_matrix (Compressed Sparse Row)
#  complex numbers handled by extra trailing dimension in hdf array

import numpy as np
from itertools import product
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

# =========================== level 2: kpt ==========================
def disp_in_box(drij, lbox=1):
  # for +/- 1 cell
  sel = drij <= lbox/2
  drij[sel] += lbox
  sel = drij > lbox/2
  drij[sel] -= lbox
  return drij

def calc_qk2k(tfracs, rtol=1e-6):
  nq = len(tfracs)
  # q+k
  qks = disp_in_box(tfracs[:, np.newaxis]+tfracs[np.newaxis, :])
  qk2k = -np.ones([nq, nq], dtype=int)
  for iq, qk in enumerate(qks):
    # find matching k in BZ
    dtvecs = disp_in_box(qk - tfracs[:, np.newaxis])
    dtmags = np.linalg.norm(dtvecs, axis=-1)
    sel = dtmags < rtol
    # record q+k->k map
    idx = np.where(sel)[1]
    qk2k[iq] = idx
  return qk2k

# =========================== level 3: ERI ==========================

def calc_pair_densities_on_fftgrid(ukl, gvl, raxes, rvecs,
                                   show_progress=False):
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
  if show_progress:
    from qharv.field import sugar
    icalc = 0
    bar = sugar.get_progress_bar(nkpt*(nkpt-1)//2+nkpt)
    print('storing pair densities')
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
      if show_progress:
        bar.update(icalc)
        icalc += 1
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

def calc_kpij_fftn(Pij, mesh):
  nstate, nstate1, ngrid = Pij.shape
  ngrid1 = np.prod(mesh)
  assert ngrid1 == ngrid
  kPij = np.zeros((nstate, nstate, ngrid), dtype=np.complex128)
  for i, j in product(range(nstate), repeat=2):
    p3d = Pij[i, j].reshape(mesh)
    val3d = np.fft.fftn(p3d)
    val1d = val3d.ravel()/np.prod(mesh)
    kPij[i, j] = val1d
  return kPij

def check_kpij(kvecs, kPij, rvecs, Pij, mesh):
  ngrid = np.prod(mesh)
  nstate = len(kPij)
  eikr = calc_eikr(kvecs, rvecs)
  for i, j in product(range(nstate), repeat=2):
    pr0 = Pij[i, j]
    pvec = kPij[i, j]
    pr1 = np.fft.ifftn(pvec.reshape(mesh)).ravel()*ngrid
    assert np.allclose(pr0, pr1, atol=1e-8)
    pr2 = np.dot(pvec, eikr)
    assert np.allclose(pr0, pr2, atol=1e-8)

def get_vg(kvecs, vol):
  ndim = kvecs.shape[1]
  if (ndim < 2) or (ndim > 3):
    msg = 'ndim %d not supported' % ndim
    raise RuntimeError(ndim)
  # 3D: 4\pi/k^2
  pre = 4*np.pi
  k2 = np.einsum('ki,ki->k', kvecs, kvecs)
  if ndim == 2:  # 2\pi/k
    pre = 2*np.pi
    k2 = k2**0.5
  coulqG = np.zeros(len(kvecs))
  zsel = abs(k2) > 1e-8
  coulqG[zsel] = pre/k2[zsel]/vol
  return coulqG

def assemble_eri(kvecs, kPli, kPjm, vol, q=None):
  if q is None:
    q = np.zeros(ndim)
  nstate00, nstate01, ngrid0 = kPli.shape
  nstate10, nstate11, ngrid1 = kPjm.shape
  assert len(kvecs) == ngrid0
  assert ngrid0 == ngrid1
  # put 1/r on reciprocal-space grid
  coulqG = get_vg(kvecs+q, vol)
  # sandwich between pair densities
  eri0 = np.zeros((nstate00, nstate01, nstate10, nstate11),
                  dtype=np.complex128)
  for i in range(nstate00):
    for l in range(nstate01):
      left = kPli[l, i].conj()*coulqG
      for j in range(nstate10):
        for m in range(nstate11):
          kpjm = kPjm[j, m]
          eri0[i, j, l, m] = np.dot(left, kpjm)
  npair0 = nstate00*nstate01
  npair1 = nstate10*nstate11
  eri0 = eri0.reshape(npair0, npair1)
  return eri0

def calc_kp_eri(iQl, tfracs, raxes, gvl, ukl, mesh, show_progress=False):
  nbndl = [len(uk) for uk in ukl]
  nbnd = nbndl[0]  # !!!! assume same nbnd at all kpts
  if not np.allclose(nbndl, nbnd):
    msg = 'kp eri for varying nbnd'
    raise NotImplementedError(msg)
  ndim = len(raxes)
  assert len(mesh) == ndim
  qktok2 = calc_qk2k(tfracs)
  nkpt = len(tfracs)
  # volume given by supercell
  axes = 2*np.pi*np.linalg.inv(raxes).T
  vol = abs(np.linalg.det(axes))*nkpt
  # unit cell defines FFT grid
  kvecs = get_kvecs(raxes, mesh)
  rvecs = get_rvecs(axes, mesh)
  ngrid = len(rvecs)
  # pre-compute all pair densities in real space
  Pijs = calc_pair_densities_on_fftgrid(ukl, gvl, raxes, rvecs,
                                        show_progress=show_progress)

  npair = nbnd**2
  tvecs = np.dot(tfracs, raxes)
  nQ = len(iQl)
  kperi = np.zeros([nQ, nkpt, nkpt, npair, npair], dtype=np.complex128)
  if show_progress:
    from qharv.field import sugar
    icalc = 0
    bar = sugar.get_progress_bar(nQ*nkpt**2)
    print('assembling KP ERI')
  for iQ in iQl:
    for ik, lk in enumerate(qktok2[iQ]):
      Qvec = tvecs[ik]-tvecs[lk]
      kvi = np.dot(gvl[ik], raxes)
      kvl = np.dot(gvl[lk], raxes)
      # put pair density on real-space FFT grid
      Pli = Pijs[lk, ik]
      # transform pair density to reciprocal-space FFT grid
      kPli = calc_kpij_fftn(Pli, mesh)
      #check_kpij(kvecs, kPli, rvecs, Pli, mesh)
      for mk, jk in enumerate(qktok2[iQ]):
        dQ = tvecs[mk]-tvecs[jk]-Qvec
        phase_jm = calc_eikr(dQ, rvecs)
        kvm = np.dot(gvl[mk], raxes)
        kvj = np.dot(gvl[jk], raxes)
        Pjm = Pijs[jk, mk]
        # compute Pij Fourier coefficients
        kPjm = calc_kpij_fftn(Pjm*phase_jm, mesh)
        # calculate ERIs
        eri0 = assemble_eri(kvecs, kPli, kPjm, vol, q=Qvec)
        kperi[iQ, ik, mk, :, :] = eri0
        if show_progress:
          bar.update(icalc)
          icalc += 1
  return kperi

def decompress_kp_eri(kperi, qk2k, nmo_pk):
  nkpt = len(nmo_pk)
  offsets = np.cumsum(nmo_pk)-nmo_pk[0]
  ntot = sum(nmo_pk)
  eri = np.zeros((ntot,)*4, dtype=kperi.dtype)
  for iq, ki, kl in product(range(nkpt), repeat=3):
    kk = qk2k[iq,ki]
    kj = qk2k[iq,kl]
    ipair = 0
    for i in range(nmo_pk[ki]):
      I = i + offsets[ki]
      for j in range(nmo_pk[kj]):
        J = j + offsets[kj]
        jpair = 0
        for k in range(nmo_pk[kk]):
          K = k + offsets[kk]
          for l in range(nmo_pk[kl]):
            L = l + offsets[kl]
            val = kperi[iq, ki, kl, ipair, jpair]
            eri[I, J, K, L] = val
        jpair += 1
    ipair += 1
  return eri
