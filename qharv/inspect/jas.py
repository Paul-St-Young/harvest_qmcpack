# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to visualize jastrow potential U. Jastrow J=exp(-U).
#
# Most function take lxml.etree.Element as first input (doc). For example:
#  doc = etree.parse(fxml)
#  doc1 = doc.find('.//wavefunction')
import numpy as np

from qharv.seed import xml
from qharv.reel import mole


def find_iopt(iopt, opt_dir):
  """ locate opt.xml file in opt_dir
  Args:
    iopt (int): optimization step (series id)
    opt_dir (str): path to optimization run
  Return:
    str: floc, location of the requested opt.xml file
  """
  st = 's'+str(iopt).zfill(3)
  regex = '*%s.opt.xml' % st
  flist = mole.files_with_regex(regex, opt_dir)
  if len(flist) != 1:
    raise RuntimeError('find %d expected 1' % len(flist))
  floc = flist[0]
  return floc


def get_coeff(doc, coef_id):
  """ extract coefficients from a <coefficients> node

  Args:
    doc (etree.Element): <coefficient> node
    coef_id (str): coefficient name (id), e.g. 'uu', 'ud', 'cG2'
  Return:
    array-like: extracted coefficients
  """
  cnode = doc.find('.//coefficients[@id="%s"]' % coef_id)
  coefs = np.array(cnode.text.split(), dtype=float)
  return coefs


def bspline_on_rgrid(doc, cid, rcut=None, cusp=None, rgrid=None):
  """ evaluate QMCPACK Basis spline on a real-space grid

  doc must contain the desired Bspline <correlation> <coefficient> nodes
  <correlation size="8" rcut="4.0">
    <coefficient id="cid" type="Array"> 0.5 0.1 </coefficients>
  </correlation>

  Args:
    doc (etree.Element): must contain the Bspline component to read
  Return:
    list: a list of floats
  """
  import jastrow  # from Mark Dewing

  cnode = doc.find('.//coefficients[@id="%s"]' % cid)
  if (rcut is None) or (cusp is None) or (rgrid is None):
    # need more info
    corr = cnode.getparent()
    if corr.tag != 'correlation':
      raise RuntimeError('found %s expected correlation' % corr.tag)
    if rcut is None:
      rcut = float(corr.get('rcut'))
    if cusp is None:
      cusp = float(corr.get('cusp'))
    if rgrid is None:
      rgrid = np.linspace(0, rcut, 64)
  # end if
  if max(rgrid) >= rcut:
    raise RuntimeError('grid exceeds rcut')

  # rcut, cusp and rgrid must be given by this point
  coefs = get_coeff(doc, cid)
  fspl = jastrow.create_jastrow_from_param(coefs, cusp, rcut)
  vals = [fspl.evaluate_v(r) for r in rgrid]
  return vals
