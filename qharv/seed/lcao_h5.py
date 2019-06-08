# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to read linear combination of atomic orbitals (LCAO) hdf5 file
from qharv.seed import xml
from qharv.seed.wf_h5 import read, ls

# ====================== level 1: extract basic info =======================

def abs_grid(fp, iabs):
  """Extract <grid> for some <atomicBasisSet>

  Args:
    fp (h5py.File): LCAO h5
  Return:
    lxml.etree.Element: <grid>
  """
  path = 'basisset/atomicBasisSet%d' % iabs
  grid = xml.etree.Element('grid')
  _add_attribs(grid, fp, path, ['type', 'ri', 'rf', 'npts'], prefix='grid_')
  return grid

def bg_radfunc(fp, iabs, ibg, irf):
  """Extract <radfunc> for some <basisGroup>

  Args:
    fp (h5py.File): LCAO h5
  Return:
    lxml.etree.Element: <radfunc>
  """
  path = 'basisset/atomicBasisSet%d/basisGroup%d' % (iabs, ibg)
  rfpath = '%s/radfunctions/DataRad%d' % (path, irf)
  rf = xml.etree.Element('radfunc')
  _add_attribs(rf, fp, rfpath, ['exponent', 'contraction'])
  return rf

def basis_group(fp, iabs, ibg):
  """Extract <basisGroup>

  Args:
    fp (h5py.File): LCAO h5
  Return:
    lxml.etree.Element: <basisGroup>
  """
  bgpath = 'basisset/atomicBasisSet%d/basisGroup%d' % (iabs, ibg)
  bg = xml.etree.Element('basisGroup')
  _add_attribs(bg, fp, bgpath, ['rid', 'n', 'l', 'type'])
  # add radial functions
  nrf = fp['%s/NbRadFunc' % bgpath][()][0]
  for irf in range(nrf):
    rf = bg_radfunc(fp, iabs, ibg, irf)
    bg.append(rf)
  return bg

def _add_attribs(node, fp, path, attribs, prefix=''):
  for attrib in attribs:
    apath = '%s/%s' % (path, prefix+attrib)
    val = fp[apath][()][0]
    node.set(attrib, str(val))

# ====================== level 2: hdf5 to xml =======================

def basisset(fp):
  """Extract <basisset>

  Args:
    fp (h5py.File): LCAO h5
  Return:
    lxml.etree.Element: <basisset>
  """
  bs = xml.etree.Element('basisset')
  bsname = fp['basisset/name'][()][0]
  bs.set('name', bsname)

  nabs = fp['basisset/NbElements'][()][0]
  for iabs in range(nabs):  # atomicBasisSet
    path = 'basisset/atomicBasisSet%d' % iabs
    myabs = xml.etree.Element('atomicBasisSet')
    abs_attribs = ['name', 'angular', 'elementType', 'normalized']
    _add_attribs(myabs, fp, path, abs_attribs)
    # !!!! make type "Gaussian", otherwise default to "Numeric"
    myabs.set('type', 'Gaussian')
    # each atomic basis set should have a <grid> and a few <basisGroup>s
    grid = abs_grid(fp, iabs)
    myabs.append(grid)
    # build basis groups
    nbg = fp['%s/NbBasisGroups' % path][()][0]
    for ibg in range(nbg):
      bg = basis_group(fp, iabs, ibg)
      myabs.append(bg)
    bs.append(myabs)
  return bs

def sposet(fp, bsname, cname, ik=0, ispin=0):
  path = 'KPTS_%d/eigenset_%d' % (ik, ispin)
  mo_coeff = fp[path][()]
  nmo, nao = mo_coeff.shape # !!!! check transpose
  # build <sposet>
  ss = xml.etree.Element('sposet')
  ss.set('basisset', bsname)
  ss.set('name', 'spo-ud')
  ss.set('size', str(nmo))
  # add <coefficient>
  cnode = xml.etree.Element('coefficient')
  cnode.set('size', str(nao))
  cnode.set('id', cname)
  cnode.text = xml.arr2text(mo_coeff)
  ss.append(cnode)
  return ss
