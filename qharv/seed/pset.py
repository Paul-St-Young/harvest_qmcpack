import numpy as np
from qharv.seed import xml

def sprinkle(centers,alpha):
  """ spinkle a group of particles around 'centers' using Gaussians having exponent 'alpha'
  Args:
   centers (np.array): centers to sprink the particles around
   alpha (float): exponent of Gaussian used to sprinkle particles, width = sqrt(1/(4*alpha))
  Returns:
   np.array: new positions
  """
  nions = len(centers)
  sig   = np.sqrt(1./(4.*alpha))
  move = np.random.randn( *np.shape(centers) )*sig
  return centers + move
# end def

def feed(pos,group,pset):
  """ feed positions to a group in particle set
  Args:
   pos (np.array): new positions
   group (str): name of particle group
   pset (lxml.etree._Element): <particleset>
  Effect:
   pset is edited
  """

  gnode = pset.find('.//group[@name="%s"]'%group)
  if gnode is None:
    pname = pset.get('name')
    msg   = 'no particle group "%s" in %s %s'%(group,pset.tag,pname)
    raise RuntimeError(msg)
  # end if

  nptcl = int(gnode.get('size'))
  npos  = len(pos)
  assert nptcl == npos

  rda = pset.get('random') # see if particleset is currently initialized at random
  if rda is not None:
    pset.attrib.pop('random')
  # end if

  pnode = gnode.find('.//attrib[@name="position"]')
  pnode.text = xml.arr2text(pos)
# end def
