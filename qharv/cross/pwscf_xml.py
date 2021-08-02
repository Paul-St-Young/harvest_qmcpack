# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate QE pwscf xml results
import numpy as np

# ========================== level 0: read ==========================
from qharv.seed.xml import read, write, parse

# ======================== level 1: KS bands ========================
def read_bands(doc):
  from qharv.seed.xml import text2arr
  bs = doc.find('.//band_structure')
  ksl = bs.findall('.//ks_energies')
  bl = []  # eval
  for ks in ksl:
    eig = ks.find('.//eigenvalues')
    evals = text2arr(eig.text, flatten=True)
    bl.append(evals)
  bands = np.array(bl)
  return bands

def read_kpoints_and_weights(doc):
  from qharv.seed.xml import text2arr
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
  from qharv.seed.xml import text2arr
  bs = doc.find('.//band_structure')
  fs = bs.find('.//fermi_energy')
  efermi = text2arr(fs.text, flatten=True)
  return efermi

def read_reciprocal_lattice(doc):
  from qharv.seed.xml import text2arr
  astruct = doc.find('.//atomic_structure')
  alat = float(astruct.get('alat'))
  blat = 2*np.pi/alat
  rlat = doc.find('.//reciprocal_lattice')
  bl = []
  for bnode in rlat:
    b1 = text2arr(bnode.text)*blat
    bl.append(b1)
  raxes = np.array(bl)
  return raxes

# ===================== level 2: KS determinant =====================
def read_gc_occupation(fpwscf_xml, eps=0):
  doc = read(fpwscf_xml)
  omatl = []
  for ef in read_efermi(doc):
    bands = read_bands(doc)
    omat = bands < ef+eps
    omatl.append(omat.astype(int))
  return omatl

