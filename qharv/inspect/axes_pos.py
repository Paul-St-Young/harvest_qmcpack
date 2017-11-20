# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to process crystal structure specified by axes,pos
import numpy as np
import pandas as pd

def abc(axes):
  """ a,b,c lattice parameters
  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    list: [a,b,c]
  """
  abc = [np.linalg.norm(vec) for vec in axes]
  return abc
# end def

def volume(axes):
  """ volume of a simulation cell
  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    float: volume of cell
  """
  return np.dot(axes[0],np.cross(axes[1],axes[2]))
# end def volume

def rs(axes,natom):
  """ rs density parameter (!!!! axes MUST be in units of bohr)
  Args:
    axes (np.array): lattice vectors in row-major, MUST be in units of bohr
  Returns:
    float: volume of cell
  """
  vol = volume(axes)
  vol_pp = vol/natom # volume per particle
  rs  = ((3*vol_pp)/(4*np.pi))**(1./3) # radius for spherical vol_pp
  return rs
# end def rs

def rins(axes):
  """ radius of the inscribed sphere inside the given cell
  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    float: radius of the inscribed sphere
  """
  a01 = np.cross(axes[0],axes[1])
  a12 = np.cross(axes[1],axes[2])
  a20 = np.cross(axes[2],axes[0])
  face_areas = [np.linalg.norm(x) for x in [a01,a12,a20]]

  # 2*rins is the height from face
  rins = volume(axes)/2./max(face_areas)
  return rins
# end def rins

def properties_from_axes_pos(axes,pos):
  """ calculate properties from axes,pos alone
   essentially the simplified/customized version of:
    pymatgen.Structure(axes,elem,pos,coords_are_cartesian=True)
  Args:
    axes (np.array): crystal lattice vectors in row-major storage
    pos (np.array):  atomic positions in row-major storage
  Returns:
    pd.Series: a list of properties
  """

  # canonical properties
  natom  = len(pos)
  vol    = volume(axes)
  vol_pp = vol/natom # volume per particle
  rs     = (3.*vol_pp/(4*np.pi))**(1./3)
  inv_axes = np.linalg.inv(axes)
  upos   = np.dot(pos,inv_axes) # fractional coordinates

  entry = {}
  name_list = ['natom','volume','vol_pp','rs','rins','inv_axes','upos']
  val_list  = [natom,vol,vol_pp,rs,rins(axes),inv_axes,upos]
  for name,val in zip(name_list,val_list):
    entry[name] = val
  # end for

  # return pd.Series(entry)

  # ad-hoc properties (scale crystal by alat)
  alat  = axes[0][0]
  uaxes = axes/alat
  pos1  = np.dot(upos,uaxes)

  name_list = ['uaxes','pos1']
  val_list  = [uaxes,pos1]
  for name,val in zip(name_list,val_list):
    entry[name] = val
  # end for

  return pd.Series(entry)
# end def properties_from_axes_pos

def mg_lattice_from_axes_pos(axes,pos):
  """ transfer lattice propertires from pymatgen.Structure """
  import pymatgen as mg
  elem = ['H']*len(pos)
  struct = mg.Structure(axes,elem,pos,coords_are_cartesian=True)
  entry = struct.as_dict()['lattice']
  return pd.Series(entry)
# end def
