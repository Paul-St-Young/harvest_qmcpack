# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to process crystal structure specified by axes,pos
import numpy as np

# ======================== level 0: axes properties =========================
def abc(axes):
  """ a,b,c lattice parameters

  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    np.array: lattice vector lengths
  """
  abc = np.linalg.norm(axes, axis=-1)
  return abc

def raxes(axes):
  """ find reciprocal lattice vectors

  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    np.array: raxes, reciprocal lattice vectors in row-major
  """
  return 2*np.pi*np.linalg.inv(axes).T

def volume(axes):
  """ volume of a simulation cell

  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    float: volume of cell
  """
  return abs(np.dot(axes[0], np.cross(axes[1], axes[2])))

def rs(axes, natom):
  """ rs density parameter (!!!! axes MUST be in units of bohr)

  Args:
    axes (np.array): lattice vectors in row-major, MUST be in units of bohr
  Returns:
    float: volume of cell
  """
  vol = volume(axes)
  vol_pp = vol/natom  # volume per particle
  rs  = ((3*vol_pp)/(4*np.pi))**(1./3)  # radius for spherical vol_pp
  return rs

def rins(axes):
  """ radius of the inscribed sphere inside the given cell

  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    float: radius of the inscribed sphere
  """
  a01 = np.cross(axes[0], axes[1])
  a12 = np.cross(axes[1], axes[2])
  a20 = np.cross(axes[2], axes[0])
  face_areas = [np.linalg.norm(x) for x in [a01, a12, a20]]

  # 2*rins is the height from face
  rins = volume(axes)/2./max(face_areas)
  return rins

def rwsc(axes, dn=1):
  """ radius of the inscribed sphere inside the real-space
  Wigner-Seitz cell of the given cell

  Args:
    axes (np.array): lattice vectors in row-major
    dn (int,optional): number of image cells to search in each
     dimension, default dn=1 searches 26 images in 3D.
  Returns:
    float: Wigner-Seitz cell radius
  """
  ndim = len(axes)
  from itertools import product
  r2imgl  = []  # keep a list of distance^2 to all neighboring images
  images = product(range(-dn, dn+1), repeat=ndim)
  for ushift in images:
    if sum(ushift) == 0:
      continue  # ignore self
    shift = np.dot(ushift, axes)
    r2imgl.append(np.dot(shift, shift))
  rimg = np.sqrt(min(r2imgl))
  return rimg/2.

def tmat(axes0, axes1):
  """ calculate the tiling matrix that takes axes0 to axes

  Args:
    axes0 (np.array): primitive cell lattice vectors in row-major
    axes1 (np.array): supercell lattice vectors in row-major
  Return:
    np.array: tmat, such that axes1 = np.dot(tmat, axes0)
  """
  return np.dot(axes1, np.linalg.inv(axes0))

# ======================== level 1: axes pos =========================
def pos_in_axes(axes, pos, ztol=1e-10):
  """ particle position(s) in cell

  Args:
    axes (np.array): crystal lattice vectors
    pos (np.array): particle position(s)
  Returns:
    pos0(np.array): particle position(s) inside the cell
  """
  upos = np.dot(pos, np.linalg.inv(axes))
  zsel = abs(upos % 1-1) < ztol
  upos[zsel] = 0
  pos0 = np.dot(upos % 1, axes)
  return pos0

def cubic_pos(nx, ndim=3):
  """ initialize simple cubic lattice in unit cube

  Args:
    nx (int) OR nxnynz (np.array): number of points along each dimension
    ndim (int): number of spatial dimensions
  Return:
    np.array: simple cubic lattice positions, shape (nx**3, ndim)
  """
  try:
    len(nx) == ndim
    nxnynz = [np.arange(n) for n in nx]
  except TypeError:
    nxnynz = [np.arange(nx)]*ndim
  pos = np.stack(
    np.meshgrid(*nxnynz, indexing='ij'),
    axis=-1
  ).reshape(-1, ndim)
  return pos

def auto_distance_table(axes, pos, dn=1):
  """ calculate distance table of a set of particles among themselves
  keep this function simple! use this to test distance_table(axes,pos1,pos2)

  Args:
    axes (np.array): lattice vectors in row-major
    pos  (np.array): particle positions in row-major
    dn (int,optional): number of neighboring cells to search in each direction
  Returns:
    np.array: dtable shape=(natom,natom), where natom=len(pos)
  """
  natom, ndim = pos.shape
  dtable = np.zeros([natom, natom], float)
  from itertools import combinations, product
  # loop through all unique pairs of atoms
  for (i, j) in combinations(range(natom), 2):  # 2 for pairs
    dists = []
    # loop through all neighboring periodic images of atom j
    #  should be 27 images for a 3D box (dn=1)
    for ushift in product(range(-dn, dn+1), repeat=ndim):
      shift = np.dot(ushift, axes)
      disp  = pos[i] - (pos[j]+shift)
      dist  = np.linalg.norm(disp)
      dists.append(dist)
    dtable[i, j] = dtable[j, i] = min(dists)
  return dtable

# ======================== level 2: advanced =========================
def displacement(axes, spos1, spos2, dn=1):
  """ single particle displacement spos1-spos2 under minimum image convention

  Args:
    axes (np.array): lattice vectors
    spos1 (np.array): single particle position 1
    spos2 (np.array): single particle position 2
    dn (int,optional): number of neighboring cells to search in each direction
  Returns:
    np.array: disp_table shape=(natom,natom,ndim)
  """
  if len(spos1) != len(spos2):
    raise RuntimeError('dimension mismatch')
  ndim = len(spos1)
  npair = (2*dn+1)**ndim  # number of images

  # find minimum image displacement
  min_disp = None
  min_dist = np.inf
  from itertools import product
  for ushift in product(range(-dn, dn+1), repeat=ndim):
    shift = np.dot(ushift, axes)
    disp  = spos1 - (spos2+shift)
    dist  = np.linalg.norm(disp)
    if dist < min_dist:
      min_dist = dist
      min_disp = disp.copy()
  return min_disp

def dimer_pairs_and_dists(axes, pos, rmax, rmin=0):
  """ find all dimers within a separtion of (rmin,rmax)

  Args:
    axes (np.array): crystal lattice vectors
    pos  (np.array): particle positions
    rmax (float): maximum dimer separation
    rmin (float,optional): minimum dimer separation
  Return:
    np.array: unique pairs, a list of (int,int) particle id pairs
    np.array: unique distances, a list of floats
  """

  # get distance table
  dtable = auto_distance_table(axes, pos)

  # locate pairs
  sel = (dtable < rmax) & (dtable > rmin)
  pairs = np.argwhere(sel)

  # remove permutation
  usel  = pairs[:, 0] < pairs[:, 1]
  upair = pairs[usel]
  udist = dtable[sel][usel]
  return upair, udist

def c_over_a(axes, cmax=True, warn=True, abtol=1e-6):
  """ calculate c/a ratio given a=b

  Args:
    axes (np.array): lattice vectors
    cmax (bool,optional): c vector is longest
  Returns:
    float: c/a
  """
  myabc = abc(axes)
  if cmax:
    cidx = np.argmax(myabc)
  else:
    cidx = np.argmin(myabc)
  aidx = (cidx+1) % 3
  bidx = (cidx+2) % 3
  if np.isclose(myabc[cidx], myabc[aidx]) or \
     np.isclose(myabc[cidx], myabc[bidx]):
    if warn:
      print('c is close to a/b; try set cmax')
  if not np.isclose(myabc[aidx], myabc[bidx], atol=abtol):
    raise RuntimeError('lattice a,b not equal')
  return myabc[cidx]/myabc[aidx]


def ase_get_spacegroup_id(axes, elem, pos, **kwargs):
  """ get space group ID using atomic simulation environment

  Args:
    axes (np.array): lattice vectors
    elem (np.array): atomic symbols
    pos  (np.array): atomic positions
  """
  from ase import Atoms
  from ase.spacegroup import get_spacegroup
  s1 = Atoms(elem, pos, cell=axes)
  sg = get_spacegroup(s1, **kwargs)
  return sg.no
