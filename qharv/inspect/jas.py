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
# ======================== level 0: file location =========================


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


# ======================== level 1: read .dat =========================


def read_dat(j2_dat):
  """ read the first two columns of a .dat file spewed out by QMCPACK

  examples:
    read_dat('J2.uu.dat')
    read_dat('uk.g000.dat')
    read_dat('BFe-e.ud.dat')

  Args:
    j2_dat (str): filename
  Return:
    tuple: (r, v), first two columns, probably (radial grid, values)
  """
  data = np.loadtxt(j2_dat)  # r v OR r v g l
  r, v = data[:, :2].T
  return r, v


# ==================== level 2: construct from xml =====================


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

def set_coeff(doc, coef_id, coefs):
  """ set coefficients for a <coefficients> node
  Args:
    doc (etree.Element): <coefficient> node
    coef_id (str): coefficient name (id), e.g. 'uu', 'ud', 'cG2'
    coefs (array): coefficients to write
  """
  cnode = doc.find('.//coefficients[@id="%s"]' % coef_id)
  cnode.text = xml.arr2text(coefs)

def bspline_on_rgrid(doc, cid, rgrid=None, rcut=None, cusp=None):
  """ evaluate QMCPACK Basis spline on a real-space grid

  doc must contain the desired Bspline <correlation> <coefficient> nodes
  <correlation size="8" rcut="4.0">
    <coefficient id="cid" type="Array"> 0.5 0.1 </coefficients>
  </correlation>

  Args:
    doc (etree.Element): must contain the Bspline component to read
    cid (str): coefficient name (id)
    rgrid (list): a list of radial grid values
    rcut (float, optional): cutoff radius, default is to read from
    <correlation>
    cusp (float, optional): cusp at r=0, default is to read from
    <correlation>
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
      try:
        cusp = float(corr.get('cusp'))
      except TypeError as err:
        if (cid == 'uu'):
          cusp = -0.25
        elif (cid == 'ud'):
          cusp = -0.5
        else:
          return err
    if rgrid is None:
      rgrid = np.linspace(0, rcut-1e-8, 64)
  # end if
  if max(rgrid) >= rcut:
    raise RuntimeError('grid exceeds rcut')

  # rcut, cusp and rgrid must be given by this point
  coefs = get_coeff(doc, cid)
  fspl = jastrow.create_jastrow_from_param(coefs, cusp, rcut)
  vals = [fspl.evaluate_v(r) for r in rgrid]
  return rgrid, vals

def make_bspline(knots, cusp, rcut):
  from functools import partial
  start = 0; stop = rcut; delta = (stop-start)/(len(knots)+1)
  delta_inv = 1./delta
  coefs = coefficients_from_knots(knots, cusp, delta)
  bsp = BsplineFunction({'ncoef': len(coefs), 'grid_start': start, 'delta_inv': delta_inv})
  return partial(bsp, {'coefs': coefs})

def coefficients_from_knots(knots, cusp, delta):
  """Prepend one knot for cusp, append three zeros for last [t^3, t^2, t, 0]"""
  coeffs = np.zeros(len(knots)+4)
  coeffs[0] = knots[1] - 2.0*delta*cusp
  coeffs[1:len(knots)+1] = knots[:]
  return coeffs

def solve_for_knots(bsp, x0, y0):
  nrow = len(x0)
  assert len(y0) == nrow
  nfit = bsp.ncoef-3
  if nrow < nfit:
    msg = 'need more than %d pts for %d parameters' % (nrow, nfit)
    raise RuntimeError(msg)
  A = np.array([bsp.derivatives(x1)[:-3] for x1 in x0])
  nrow, ncol = A.shape
  ret = np.linalg.lstsq(A, y0, rcond=None)
  coefs = ret[0]
  delta = 1./bsp.dxinv
  # convert back
  knots = coefs[1:]
  cusp = (knots[1]-coefs[0])/(2*delta)
  return knots, cusp

class BsplineFunction:
  def __init__(self, data):
    self.ncoef = data['ncoef']
    self.start = data['grid_start']
    self.dxinv = data['delta_inv']
    self.coefs = np.zeros(self.ncoef)
    self.amat = np.array([
      [-1./6,  3./6, -3./6, 1./6],
      [ 3./6, -6./6,  0./6, 4./6],
      [-3./6,  3./6,  3./6, 1./6],
      [ 1./6,  0./6,  0./6, 0./6],
    ])

  def get_ticks(self, x1):
    # location on grid
    x = x1 - self.start
    # index on grid
    u = x1 * self.dxinv
    i = int(u)  # tick index
    t = u%1  # remainder
    # ticks
    tp = [t**(3-n) for n in range(4)]
    return tp, i

  def derivatives(self, x1):
    derivs = np.zeros(self.ncoef)
    tp, i = self.get_ticks(x1)
    for j in range(4):
      derivs[i+j] = np.dot(self.amat[j], tp)
    return derivs

  def __call__(self, params, x1):
    self.coefs[:] = params['coefs']
    tp, i = self.get_ticks(x1)
    cs = self.coefs[i:i+4]
    val = cs@(self.amat@tp)
    return val
