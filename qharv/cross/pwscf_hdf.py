# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate QE 6.8 pwscf hdf results
import h5py
import numpy as np

# ========================== level 0: read ==========================
def read_save_hdf(fh5, name='evc', xname='MillerIndices'):
  """Read a wavefunction or charge density file in pwscf.save/
  Data values are assumed to be complex.

  Args:
    fh5 (str): "wfc[#].hdf5" or "charge-density.hdf5"
    name (str, optional): column name, default "evc"
    xname (str, optional): label name, default "MillerIndices"
  Return:
    x, y: x is "MillerIndices" by default, y is "evc" by default
  Example:
    >>> gvs, rhog = read_save_hdf("charge-density.hdf5", name="rhotot_g")
  """
  fp = h5py.File(fh5, 'r')
  gvc = fp[xname][()]
  evc = fp[name][()].view(np.complex128)
  fp.close()
  return gvc, evc

def find_wfc(fxml):
  """Find wfc hdf files using xml <band_structure> as guide.

  Args:
    fxml (str): "pwscf.xml"
  Return:
    list: locations of all wfc hdf files
  Example:
    >>> flist = find_wfc("pwscf.xml")
  """
  import os
  from qharv.cross import pwscf_xml
  prefix = fxml[:fxml.rfind('.')]
  dsave = prefix + '.save'
  if not os.path.isdir(dsave):
    msg = 'wfc save "%s" not found' % dsave
    raise RuntimeError(msg)
  # determine lsda
  doc = pwscf_xml.read(fxml)
  bgrp = doc.find('.//band_structure')
  lsda = pwscf_xml.read_true_false(doc, 'lsda')
  nk = int(bgrp.find('.//nks').text)
  wfcs = []
  if lsda:
    for spin in ['up', 'dw']:
      wfcs += [os.path.join(dsave, 'wfc%s%d.hdf5' % (spin, ik+1))
        for ik in range(nk)]
  else:
    wfcs += [os.path.join(dsave, 'wfc%d.hdf5' % (ik+1)) for ik in range(nk)]
  # check wfc
  missing = False
  nfound = 0
  nexpect = len(wfcs)
  for floc in wfcs:
    if not os.path.isfile(floc):
      print('%s not found' % floc)
      missing = True
    else:
      nfound += 1
  if missing:
    msg = 'found %d/%d wfc' % (nfound, nexpect)
    raise RuntimeError(msg)
  return wfcs

def read_wfc(fxml):
  from qharv.cross import pwscf_xml
  doc = pwscf_xml.read(fxml)
  bgrp = doc.find('.//band_structure')
  lsda = pwscf_xml.read_true_false(doc, 'lsda')
  flist = find_wfc(fxml)
  rets = [read_save_hdf(floc) for floc in flist]
  if lsda:  # concatenate spin up, spin dn wfc
    nkpt = len(flist)//2
    gvl = []
    evl = []
    for ik in range(nkpt):
      iup = ik
      idn = nkpt+ik
      gvup, evup = rets[iup]
      gvdn, evdn = rets[idn]
      assert np.allclose(gvup, gvdn)
      gvl.append(gvup)
      ev = np.concatenate([evup, evdn], axis=0)
      evl.append(ev)
  else:
    gvl = [ret[0] for ret in rets]
    evl = [ret[1] for ret in rets]
  return gvl, evl

def split_evc(evc, nspin):
  """Extract spin up and dn components

  Args:
    evc (np.array): shape (nbnd, npw), (2*nbnd, npw), (nbnd, 2*npw) for
      nspin = 1, 2, 4, respectively
    nspin (int): 1 (restricted), 2 (collinear), 4 (noncolin)
  Return:
    tuple: (evup, evdn), both have shape (nbnd, npw)
  """
  if nspin == 4:
    nbnd, npw2 = evc.shape
    npw = npw2//2
    evup = evc[:, :npw]
    evdn = evc[:, npw:]
  elif nspin == 2:
    nbnd2, npw = evc.shape
    nbnd = nbnd2//2
    evup = evc[:nbnd, :]
    evdn = evc[nbnd:, :]
  elif nspin == 1:
    evup = evdn = evc
  else:
    msg = 'unknown nspin=%d' % nspin
    raise RuntimeError(msg)
  return evup, evdn

# ========================= level 1: orbital ========================

def kinetic_energy(raxes, kfracs, gvl, evl, wtl):
  nkpt = len(kfracs)
  tkin_per_kpt = np.zeros(nkpt)
  for ik, (kfrac, gvs, evc, wts) in enumerate(zip(kfracs, gvl, evl, wtl)):
    kvecs = np.dot(gvs+kfrac, raxes)
    npw = len(kvecs)
    k2 = np.einsum('ij,ij->i', kvecs, kvecs)
    p2 = (evc.conj()*evc).real
    nk = np.dot(wts, p2)  # sum occupied bands for n(k)
    if len(nk) == 2*npw:  # noncolin
      tkin_per_kpt[ik] = np.dot(k2, nk[:npw]) + np.dot(k2, nk[npw:])
    else:
      tkin_per_kpt[ik] = np.dot(k2, nk)
  return tkin_per_kpt

def calc_kinetic(fxml, gvl=None, evl=None, wtl=None, lam=0.5):
  #lam = 1./2  # Hartree atomic units T = -lam*\nabla^2
  from qharv.cross import pwscf_xml
  doc = pwscf_xml.read(fxml)
  raxes = pwscf_xml.read_reciprocal_lattice(doc)
  kfracs = pwscf_xml.read_kfractions(doc)
  if wtl is None:
    wtl = pwscf_xml.read_occupations(doc)
  if (gvl is None) or (evl is None):
    gvl, evl = read_wfc(fxml)
  tkin_per_kpt = kinetic_energy(raxes, kfracs, gvl, evl, wtl)
  tkin = lam*tkin_per_kpt.mean()
  return tkin

# ========================== level 2: FFT ===========================

class FFTMesh:
  def __init__(self, mesh, dtype=np.complex128):
    self.mesh = mesh
    self.nnr = np.prod(mesh)
    self.grid = np.zeros(mesh, dtype=dtype)
  def invfft(self, gvectors, psik):
    self.grid.fill(0)
    for g, e in zip(gvectors, psik):
      self.grid[tuple(g)] = e
    psir = np.fft.ifftn(self.grid)*self.nnr
    return psir
  def fwdfft(self, gvectors, psir):
    self.grid = np.fft.fftn(psir.reshape(self.mesh))/self.nnr
    psik = np.zeros(len(gvectors), dtype=self.grid.dtype)
    for i, g in enumerate(gvectors):
      psik[i] = self.grid[tuple(g)]
    return psik

def rho_of_r(mesh, gvl, evl, wtl, wt_tol=1e-8, npol=1):
  rhor = np.zeros(mesh)
  fft = FFTMesh(mesh)
  psir = np.zeros(mesh, dtype=np.complex128)
  nkpt = len(gvl)
  for gvs, evc, wts in zip(gvl, evl, wtl):  # kpt loop
    sel = wts >= wt_tol
    npw = len(gvs)
    for ev, wt in zip(evc[sel], wts[sel]):  # bnd loop
      for ipol in range(npol):
        psir = fft.invfft(gvs, ev[ipol*npw:(ipol+1)*npw])
        r1 = (psir.conj()*psir).real
        rhor += wt*r1
  return rhor/nkpt

def calc_rhor(fxml, mesh=None, gvl=None, evl=None, wtl=None, spin_resolved=False):
  from qharv.cross import pwscf_xml
  doc = pwscf_xml.read(fxml)
  noncolin = pwscf_xml.read_true_false(doc, 'noncolin')
  if noncolin:
    npol = 2
  else:
    npol = 1
  if mesh is None:
    mesh = pwscf_xml.read_fft_mesh(doc)
  if wtl is None:
    wtl = pwscf_xml.read_occupations(doc)
  if (gvl is None) or (evl is None):
    gvl, evl = read_wfc(fxml)
  if spin_resolved:
    lsda = pwscf_xml.read_true_false(doc, 'lsda')
    if lsda:
      evupl = [ev[:len(ev)//2] for ev in evl]
      wtupl = [wt[:len(wt)//2] for wt in wtl]
      rhor_up = rho_of_r(mesh, gvl, evupl, wtupl)
      evdnl = [ev[len(ev)//2:] for ev in evl]
      wtdnl = [wt[len(wt)//2:] for wt in wtl]
      rhor_dn = rho_of_r(mesh, gvl, evdnl, wtdnl)
      return rhor_up, rhor_dn
    elif noncolin:
      rhor = rho_of_r(mesh, gvl, evl, wtl, npol=npol)
      mags = mag_of_r(mesh, gvl, evl, wtl)
      return np.r_[rhor[np.newaxis], mags]
    else:
      msg = 'cannot calculate spin-resolved density for lsda=%s' % lsda
      msg += ' and noncolin=%s' % noncolin
      raise RuntimeError(msg)
  rhor = rho_of_r(mesh, gvl, evl, wtl, npol=npol)
  return rhor

def mag_of_r(mesh, gvl, evl, wtl, wt_tol=1e-8):
  mags = np.zeros([3, *mesh])
  fft = FFTMesh(mesh)
  nkpt = len(gvl)
  for gvs, evc, wts in zip(gvl, evl, wtl):  # kpt loop
    npw = len(gvs)
    npol = evc.shape[1]//npw
    sel = wts >= wt_tol
    for ev, wt in zip(evc[sel], wts[sel]):  # bnd loop
      psia = ev[:npw]
      psib = ev[npw:]
      pra = fft.invfft(gvs, psia)
      prb = fft.invfft(gvs, psib)
      mags += wt*mag3d(pra, prb)
  return mags/nkpt

def mag3d(pra, prb):
  """Compute magnetization of spinor by contracting alpha and beta components
   with Pauli matrices. Each component has one complex value per grid point.
  Essentially reimplements sum_band.f90::get_rho_domag.

  Args:
    pra (array): shape (nnr,), up i.e. alpha component
    prb (array): shape (nnr,), dn i.e. beta component
  Return:
    array: shape (3, nnr), x, y, z components of magnetization
  """
  ab = pra.conj()*prb
  ba = prb.conj()*pra
  mags = np.array([
    (ab+ba).real,
    (1j*(ba-ab)).real,
    (pra.conj()*pra-prb.conj()*prb).real
  ])
  return mags

def psi_from_mag3d(psir, theta, phi):
  """Inverse of mag3d, after choosing gauge such that a=a^*

  Args:
    psir (float): wavefunction magnitude
    theta (float): spin polar angle
    phi (float): spin azimuthal angle
  Return:
    (float, float): (pra, prb), Sz up and dn components
  """
  pra = psir*np.cos(theta/2)
  prb = pra*np.sin(theta)*np.exp(1j*phi)
  return pra, prb

def site_resolved_magnetization(rho, pointlist, factlist):
  """Compute site-resolved magnetization for each magnetic site.
  Essentially reimplements get_locals.f90

  Args:
    rho (array): shape (nspin, *mesh), mesh is the FFT mesh w/ nnr grid points.
    pointlist (array): shape (nnr,), integers from 0 to nat+1. pointlist
     assigns each FFT grid point to an atom. 0 means not assigned.
    factlist (array): shape (nnr,), floats from 0 to 1. Weight of each point.
  Return:
    array: shape (nat, nspin), magnetization on each site
  """
  nat = np.unique(pointlist).max()
  nspin = len(rho)
  mesh = rho.shape[1:]
  nnr = np.prod(mesh)
  mags = np.zeros([nat, nspin])
  for iat in range(1, nat+1):
    sel = pointlist == iat
    rsum = [np.dot(factlist[sel], rho[ispin, sel])/nnr for ispin in range(nspin)]
    mags[iat-1, :] = rsum
  return mags

def site_resolve(rho, pointlist, factlist=None):
  """Compute density integrals for each site
  supercedes site_resolved_magnetization

  Args:
    rho (array): shape (nspin, *mesh), mesh is the FFT mesh w/ nnr grid points.
    pointlist (array): shape (nnr,), integers from 0 to nsite. pointlist
     assigns each FFT grid point to an atom. 0 means not assigned.
    factlist (array): shape (nnr,), floats from 0 to 1. Weight of each point.
  Return:
    array: shape (nsite,), integral on each site
  """
  if factlist is None:
    factlist = np.ones(len(pointlist))
  nsite = pointlist.max()
  nnr = len(pointlist)
  mags = np.zeros(nsite)
  for i in range(nsite):
    sel = pointlist == i+1
    mags[i] = np.dot(factlist[sel], rho[sel])
  return mags/nnr

def site_mag(rhos, axes, pos, mesh):
  from qharv.seed import hamwf_h5
  from qharv.inspect import axes_pos
  rvecs = hamwf_h5.get_rvecs(axes, mesh)
  pointlist = axes_pos.rcut_partition(axes, pos, rvecs)
  mags = np.array([
    site_resolve(rhor.ravel(), pointlist)
  for rhor in rhos])
  return mags
