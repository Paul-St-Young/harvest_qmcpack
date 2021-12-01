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
    for spin in ['up', 'dn']:
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
  flist = find_wfc(fxml)
  rets = [read_save_hdf(floc) for floc in flist]
  gvl = [ret[0] for ret in rets]
  evl = [ret[1] for ret in rets]
  return gvl, evl

# ========================= level 1: orbital ========================

def kinetic_energy(raxes, kfracs, gvl, evl, wtl):
  nk = len(kfracs)
  tkin_per_kpt = np.zeros(nk)
  for ik, (kfrac, gvs, evc, wts) in enumerate(zip(kfracs, gvl, evl, wtl)):
    kvecs = np.dot(gvs+kfrac, raxes)
    k2 = np.einsum('ij,ij->i', kvecs, kvecs)
    p2 = (evc.conj()*evc).real
    nk = np.dot(wts, p2)  # sum occupied bands for n(k)
    tkin_per_kpt[ik] = np.dot(k2, nk)
  return tkin_per_kpt

def calc_kinetic(fxml, gvl=None, evl=None, lam=0.5):
  #lam = 1./2  # Hartree atomic units T = -lam*\nabla^2
  from qharv.cross import pwscf_xml
  if (gvl is None) or (evl is None):
    gvl, evl = read_wfc(fxml)
  doc = pwscf_xml.read(fxml)
  raxes = pwscf_xml.read_reciprocal_lattice(doc)
  kfracs = pwscf_xml.read_kfractions(doc)
  omat = pwscf_xml.read_occupations(doc)
  tkin_per_kpt = kinetic_energy(raxes, kfracs, gvl, evl, omat)
  tkin = lam*tkin_per_kpt.sum()
  return tkin
