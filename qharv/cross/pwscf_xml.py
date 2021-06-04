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
  bl = []
  for ks in ksl:
    eig = ks.find('.//eigenvalues')
    evals = text2arr(eig.text, flatten=True)
    bl.append(evals)
  bands = np.array(bl)
  return bands

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
