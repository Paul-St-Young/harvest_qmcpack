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

def read_value(node, name, dtype=float):
  child = node.find('.//%s' % name)
  text = child.text
  val = dtype(text)
  return val

# ==================== level 1: atomic structure ====================
def read_alat(doc):
  astruct = doc.find('.//atomic_structure')
  alat = float(astruct.get('alat'))
  return alat

def read_cell(doc):
  astruct = doc.find('.//atomic_structure')
  cell = astruct.find('.//cell')
  al = []
  for anode in cell:
    a1 = text2arr(anode.text)
    al.append(a1)
  axes = np.array(al)
  return axes

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

def read_elem_pos(doc):
  astruct = doc.find('.//atomic_structure')
  atoms = astruct.findall('.//atom')
  elem = []
  pos = []
  for atom in atoms:
    e = atom.get('name')
    elem.append(e)
    p = text2arr(atom.text)
    pos.append(p)
  return np.array(elem), np.array(pos)

# ====================== level 1: basic outputs =====================
def read_magnetization(doc):
  node = doc.find('.//magnetization')
  ret = dict()
  for name in ['total', 'absolute']:
    ret[name] = read_value(node, name)
  return ret

def read_total_energy(doc):
  node = doc.find('.//total_energy')
  names = ['etot', 'eband', 'ehart', 'vtxc', 'etxc', 'ewald', 'demet']
  ret = dict()
  for name in names:
    ret[name] = read_value(node, name)
  return ret

# ======================== level 2: KS bands ========================
def read_bands(doc):
  # !!!! this concatenates up- and dn-spin bands
  bs = doc.find('.//band_structure')
  if bs is None:
    bs = doc
  lsda = read_true_false(bs, 'lsda')
  ksl = bs.findall('.//ks_energies')
  bl = []  # eval
  for ks in ksl:
    eig = ks.find('.//eigenvalues')
    evals = text2arr(eig.text, flatten=True)
    if lsda:
      evals = evals.reshape(2, -1)
    bl.append(evals)
  bands = np.array(bl)
  return bands

def read_occupations(doc):
  bs = doc.find('.//band_structure')
  if bs is None:
    bs = doc
  lsda = read_true_false(bs, 'lsda')
  noncolin = read_true_false(bs, 'noncolin')
  ocl = bs.findall('.//occupations')
  ol = []
  for oc in ocl:
    o1 = text2arr(oc.text, flatten=True)
    if noncolin:
      ol.append(o1)
    elif lsda:
      ol.append(o1.reshape(2, -1))
    else:  # restricted
      ol.append([o1, o1])
  omat = np.array(ol)
  return omat

def read_kpoints_and_weights(doc):
  bs = doc.find('.//band_structure')
  if bs is None:
    bs = doc
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
  if bs is None:
    bs = doc
  fs = bs.find('.//fermi_energy')
  efermi = text2arr(fs.text, flatten=True)
  return efermi

def read_kfractions(doc):
  kpts, wts = read_kpoints_and_weights(doc)
  alat = read_alat(doc)
  blat = 2*np.pi/alat
  raxes = read_reciprocal_lattice(doc)
  kfracs = np.dot(kpts*blat, np.linalg.inv(raxes))
  return kfracs

def sum_band(doc, read=False):
  """Sum eigenvalues of occupied orbitals

  Args:
    doc (etree.Element): <qes:espresso> or anything with <band_structure>
  Return:
    float: sum of weighted eigenvalues
  Example:
    >>> doc = pwscf_xml.read('pwscf.xml')
    >>> doc.find('.//band_structure')
    >>> e1 = pwscf_xml.sum_band(bgrp)
  """
  bgrp = doc.find('.//band_structure')
  if bgrp is None:
    bgrp = doc
  if read:
    enode = doc.find('.//eband')
    eband = float(enode.text)
  else:
    bands = read_bands(bgrp)
    omat = read_occupations(bgrp)
    nkpt, nbnd = omat.shape
    eband = np.dot(bands.ravel(), omat.ravel())/nkpt
  return eband

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

