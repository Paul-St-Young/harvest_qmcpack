# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to inspect and validate a QMCPACK input

from qharv.seed import xml
from qharv.inspect import axes_pos

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
    rins = axes_pos.rins(axes)
    print rins
    for corr in corrs:
      rc = rcut(corr)
      print rc
      if (rc<=0) or (rc>rins):
        valid = False
      # end if
    # end for corr
  else:
    raise NotImplementedError('deal with bconds="%s"'%bc_str)
  # end if

  return valid
# end def validate_bspline_rcut
