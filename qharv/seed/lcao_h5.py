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
  for attrib in ['type', 'ri', 'rf', 'npts']:
    gpath = '%s/grid_%s' % (path, attrib)
    val = fp[gpath][()][0]
    grid.set(attrib, str(val))
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
  for attrib in ['exponent', 'contraction']:
    apath = '%s/%s' % (rfpath, attrib)
    val = fp[apath][()][0]
    rf.set(attrib, str(val))
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
  for attrib in ['rid', 'n', 'l', 'type']:
    apath = '%s/%s' % (bgpath, attrib)
    val = fp[apath][()][0]
    bg.set(attrib, str(val))
  # add radial functions
  nrf = fp['%s/NbRadFunc' % bgpath][()][0]
  for irf in range(nrf):
    rf = bg_radfunc(fp, iabs, ibg, irf)
    bg.append(rf)
  return bg

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
    for attrib in ['name', 'angular', 'elementType', 'normalized']:
      apath = '%s/%s' % (path, attrib)
      val = fp[apath][()][0]
      myabs.set(attrib, str(val))
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
