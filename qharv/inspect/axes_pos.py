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

def raxes(axes):
  """ find reciprocal lattice vectors 
  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    np.array: raxes, reciprocal lattice vectors in row-major
  """
  a1,a2,a3 = axes
  vol = volume(axes)

  b1 = 2*np.pi*np.cross(a2, a3)/vol
  b2 = 2*np.pi*np.cross(a3, a1)/vol
  b3 = 2*np.pi*np.cross(a1, a2)/vol
  return np.array([b1,b2,b3])
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

def rwsc(axes,dn=1):
  """ radius of the inscribed sphere inside the real-space Wigner-Seitz cell of the given cell
  Args:
    axes (np.array): lattice vectors in row-major
    dn (int,optional): number of image cells to search in each dimension, default dn=1 searches 26 images in 3D.
  Returns:
    float: Wigner-Seitz cell radius
  """
  ndim = len(axes)
  from itertools import product
  r2imgl  = [] # keep a list of distance^2 to all neighboring images
  images = product(range(-dn,dn+1),repeat=ndim)
  for ushift in images:
    if sum(ushift)==0: continue # ignore self
    shift = np.dot(ushift,axes)
    r2imgl.append( np.dot(shift,shift) )
  # end for
  # find minimum image distance
  rimg = np.sqrt( min(r2imgl) )
  return rimg/2.
# def rwsc

def auto_distance_table(axes,pos,dn=1):
  """ calculate distance table of a set of particles among themselves
   keep this function simple! use this to test distance_table(axes,pos1,pos2)
  Args:
    axes (np.array): lattice vectors in row-major
    pos  (np.array): particle positions in row-major
    dn (int,optional): number of neighboring cells to search in each direction
  Returns:
    np.array: dtable shape=(natom,natom), where natom=len(pos)
  """
  natom,ndim = pos.shape
  dtable = np.zeros([natom,natom],float)
  from itertools import combinations,product
  # loop through all unique pairs of atoms
  for (i,j) in combinations(range(natom),2): # 2 for pairs
    dists = []
    images = product(range(-dn,dn+1),repeat=ndim) # check all neighboring images
    # loop through all neighboring periodic images of atom j
    #  should be 27 images for a 3D box (dn=1)
    for ushift in images:
      shift = np.dot(ushift,axes)
      disp  = pos[i] - (pos[j]+shift)
      dist  = np.linalg.norm(disp)
      dists.append(dist)
    # end for ushift
    dtable[i,j] = min(dists)
    dtable[j,i] = min(dists)
  # end for (i,j)
  return dtable
# end def auto_distance_table

def properties_from_axes_pos(axes,pos):
  """ calculate properties from axes,pos alone; essentially the simplified/customized version of: pymatgen.Structure(axes,elem,pos,coords_are_cartesian=True)
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
