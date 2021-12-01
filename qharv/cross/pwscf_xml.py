# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate QE pwscf xml results
import numpy as np

# ========================== level 0: read ==========================
from qharv.seed.xml import read, write, parse, text2arr

def read_true_false(node, name):
  child = node.find('.//%s' % name)
  text = child.text
  tf = None
  if text == 'true':
    tf = True
  elif text == 'false':
    tf = False
  else:
    msg = '"%s" is neither true nor false' % text
    raise RuntimeError(msg)
  return tf

# ======================== level 1: KS bands ========================
def read_bands(doc):
  # !!!! this concatenates up- and dn-spin bands
  bs = doc.find('.//band_structure')
  ksl = bs.findall('.//ks_energies')
  bl = []  # eval
  for ks in ksl:
    eig = ks.find('.//eigenvalues')
    evals = text2arr(eig.text, flatten=True)
    bl.append(evals)
  bands = np.array(bl)
  return bands

def read_occupations(doc):
  bs = doc.find('.//band_structure')
  ocl = bs.findall('.//occupations')
  ol = []
  for oc in ocl:
    o1 = text2arr(oc.text)
    ol.append(o1)
  omat = np.array(ol)
  return omat

def read_kpoints_and_weights(doc):
  bs = doc.find('.//band_structure')
  ksl = bs.findall('.//ks_energies')
  kl = []  # kpoint
  wl = []  # weight
  for ks in ksl:
    kp = ks.find('.//k_point')
    kv = text2arr(kp.text)
    kl.append(kv)
    wt = float(kp.get('weight'))
    wl.append(wt)
  return np.array(kl), np.array(wl)

def read_efermi(doc):
  bs = doc.find('.//band_structure')
  fs = bs.find('.//fermi_energy')
  efermi = text2arr(fs.text, flatten=True)
  return efermi

def read_alat(doc):
  astruct = doc.find('.//atomic_structure')
  alat = float(astruct.get('alat'))
  return alat

def read_reciprocal_lattice(doc):
  alat = read_alat(doc)
  blat = 2*np.pi/alat
  rlat = doc.find('.//reciprocal_lattice')
  bl = []
  for bnode in rlat:
    b1 = text2arr(bnode.text)*blat
    bl.append(b1)
  raxes = np.array(bl)
  return raxes

def read_kfractions(doc):
  kpts, wts = read_kpoints_and_weights(doc)
  alat = read_alat(doc)
  blat = 2*np.pi/alat
  raxes = read_reciprocal_lattice(doc)
  kfracs = np.dot(kpts*blat, np.linalg.inv(raxes))
  return kfracs

def sum_band(bgrp):
  """Sum eigenvalues of occupied orbitals

  Args:
    bgrp (etree.Element): <band_structure>
  Return:
    float: one-body energy
  Example:
    >>> doc = pwscf_xml.read('pwscf.xml')
    >>> doc.find('.//band_structure')
    >>> e1 = pwscf_xml.sum_band(bgrp)
  """
  ksl = bgrp.findall('.//ks_energies')
  e1 = 0
  for ks in ksl:
    egrp = ks.find('.//eigenvalues')
    evals = text2arr(egrp.text, flatten=True)
    ogrp = ks.find('.//occupations')
    occs = text2arr(ogrp.text, flatten=True)
    eks = np.dot(evals, occs)
    e1 += eks
  return e1

# ======================= level 1: meta data ========================
def read_fft_mesh(doc):
  node = doc.find('.//fft_grid')
  nr1 = int(node.get('nr1'))
  nr2 = int(node.get('nr2'))
  nr3 = int(node.get('nr3'))
  mesh = np.array([nr1, nr2, nr3])
  return mesh

# ===================== level 2: KS determinant =====================
def read_gc_occupation(fpwscf_xml, eps=0):
  doc = read(fpwscf_xml)
  omatl = []
  for ef in read_efermi(doc):
    bands = read_bands(doc)
    omat = bands < ef+eps
    omatl.append(omat.astype(int))
  return omatl

