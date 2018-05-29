# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to inspect and validate a QMCPACK input
#  first argument is generally an lxml.Element

import os
import numpy as np
from qharv.seed import xml
from qharv.inspect import axes_pos


def check_wfh5_access(doc, calc_dir):
  """ check that the Bspline h5 file can be accessed from input location

  Args:
    doc (lxml.Element): must containt <sposet_builder>
    calc_dir (str): directory to contain the QMCPACK input (doc)
  Returns:
    bool: can access all h5 files
  """

  access = True

  # not compatible with legacy input
  detset = doc.find('.//determinantset[@type="bspline"]')
  if detset is not None:
    raise RuntimeError('legacy input')

  # check all Bspline orbital builders
  bb_list = doc.findall('.//sposet_builder[@type="bspline"]')
  for bb in bb_list:
    # get relative wf hdf5 location
    href = bb.get('href')

    # get absolute wf hdf5 location
    wf_h5_floc = os.path.abspath( os.path.join(calc_dir,href) )

    # check accessibility
    if not os.path.isfile(wf_h5_floc):
      access = False
  return access


def rcut(corr):
  rca = corr.get('rcut')
  if rca is None:
    raise RuntimeError('no rcut found in %s'%xml.str_rep(corr))
  # end if
  rc = float(rca)
  return rc
# end def rcut

def validate_bspline_rcut(node,ignore_empty=False):
  """ check that 1D bspline functions have valid cutoff radius

  Args:
    node (lxml.etree._Element): xml node containing <simulationcell> and <correlation>.
    ignore_empty (bool,optional): ignore inputs without <correlation>, default=False
  Returns:
    bool: valid input
  """

  # see if boundary condition is understandable
  bconds = node.find('.//parameter[@name="bconds"]')
  bc_str = ''.join( bconds.text.split() ).strip()
  if bc_str not in set(['nnn','ppp']):
    raise NotImplementedError('deal with "%s" slabs some other day'%bc_str)
  # end if

  # make a list of nodes to check
  corrs = node.findall('.//correlation')
  if len(corrs) == 0:
    if ignore_empty: return True
    raise RuntimeError('no <correlation> node to check, set ignore_empty=True to continue') 
  # end if

  valid = True
  if bc_str == 'nnn': # !!!! un-tested
    for corr in corrs: # rcut must exist and be positive
      rc = rcut(corr)
      if rc<=0:
        valid = False
      # end if
    # end for corr
  elif bc_str == 'ppp':
    axes = xml.get_axes(node)
    #rsc = axes_pos.rins(axes) # radius of inscribed sphere in simulation cell
    rwsc = axes_pos.rwsc(axes) # radius of inscribed sphere in Wigner-Seitz cell
    # Wigner-Seitz cell radius is the more optimal minimum for correlation functions
    for corr in corrs:
      try: # rcut may not be provided in periodic simulation
        rc = rcut(corr)
      except RuntimeError as err:
        not_found = 'no rcut found' in str(err)
        if not_found:
          pass # let QMCPACK decide
        else:
          raise err # pass on unknown error
        # end if
      # end try
      if (rc<=0) or (rc>rwsc):
        print('rcut for %s = %3.4f is invaid. note: rwsc = %3.4f.' % (corr.get('id'),rc,rwsc) )
        valid = False
      # end if
    # end for corr
  else:
    valid = False
    raise NotImplementedError('deal with bconds="%s"'%bc_str)
  # end if

  return valid
# end def validate_bspline_rcut

def check_wf_hdf5(snode,calc_dir,folded):
  """ check that spline single-particle orbitals are defined in the correct 
  simulation cell. snode must hold the simulation cell and basis builder with 
  a reference to the wf hdf5.

  Args:
    snode (lxml.Element): e.g. parsed <qmcsystem>
    calc_dir (str): directory to contain the QMCPACK input
    folded (bool): True, if DFT was performed in unit cell and orbitals are tiled to super cell
  Returns:
    bool: consistent
  """

  # step 1: get wf hdf5 location
  bb_list = snode.findall('.//sposet_builder[@href]')
  if not (len(bb_list) == 1):
    raise RuntimeError('wrong number of sposet_builder %d'%len(bb_list))
  # end if
  bb = bb_list[0]
  href = bb.get('href')
  wf_h5_floc = os.path.abspath( os.path.join(calc_dir,href) )
  if not os.path.isfile(wf_h5_floc):
    raise RuntimeError('failed to find wf hdf5 %s'%wf_h5_floc)
  # end if

  # step 2: get axes from wf hdf5
  from qharv.seed import wf_h5
  fp = wf_h5.read(wf_h5_floc)
  h5_axes = wf_h5.get(fp,'axes')
  fp.close()

  # step 3: get axes from xml input
  sc_node = snode.find('.//simulationcell')
  inp_axes = xml.text2arr( sc_node.find('.//parameter[@name="lattice"]').text )

  # step 4: compare
  consistent = False
  # assume folded = False
  if folded:
    raise NotImplementedError('lattice check is not implemented for folded structure')
  else:
    consistent = np.allclose(h5_axes,inp_axes)
  # end if
  return consistent
# end def

