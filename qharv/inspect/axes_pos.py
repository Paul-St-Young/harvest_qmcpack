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
  return abs(np.linalg.det(axes))

def rs(axes, natom):
  """ rs density parameter (!!!! axes MUST be in units of bohr)

  Args:
    axes (np.array): lattice vectors in row-major, MUST be in units of bohr
  Returns:
    float: volume of cell
  """
  vol = volume(axes)
  vol_pp = vol/natom  # volume per particle
  ndim = len(axes)
  # PRB 84, 115115 (2011).
  rs = ((2*(ndim-1)*np.pi)/(ndim*vol_pp))**(-1./ndim)
  return rs

def rins(axes):
  """ radius of the inscribed sphere inside the given cell

  Args:
    axes (np.array): lattice vectors in row-major
  Returns:
    float: radius of the inscribed sphere
  """
  ndim = len(axes)
  if ndim == 3:
    a01 = np.cross(axes[0], axes[1])
    a12 = np.cross(axes[1], axes[2])
    a20 = np.cross(axes[2], axes[0])
    face_areas = [np.linalg.norm(x) for x in [a01, a12, a20]]
  elif ndim == 2:
    face_areas = abc(axes)
  else:
    msg = 'cannot calculate rins of ndim=%d' % ndim
    raise RuntimeError(msg)
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
  """ calculate the tiling matrix that takes axes0 to axes1

  Args:
    axes0 (np.array): primitive cell lattice vectors in row-major
    axes1 (np.array): supercell lattice vectors in row-major
  Return:
    np.array: tmat, such that axes1 = np.dot(tmat, axes0)
  """
  return np.dot(axes1, np.linalg.inv(axes0))

def rtmat(raxes0, raxes1):
  """ calculate the tiling matrix that takes reciprocal raxes0 to raxes1

  Args:
    raxes0 (np.array): primitive cell reciprocal lattice vectors in row-major
    raxes1 (np.array): supercell reciprocal lattice vectors in row-major
  Return:
    np.array: tilematrix responsible for lattices with rec. raxes0 to raxes1
  """
  return np.dot(raxes0, np.linalg.inv(raxes1)).T

def sum_lattice(f, raxes, kc):
  from qharv.seed.hamwf_h5 import get_ksphere
  kvecs = get_ksphere(raxes, kc)[1:]  # exclude 0
  k = np.linalg.norm(kvecs, axis=-1)
  fk = f(k)
  return fk.sum()

def madelung(axes, nr=3, rckc=30.0):
  from scipy.special import erfc
  amat = axes
  ndim = len(amat)
  rc = rwsc(amat)*nr
  kc = rckc/rc
  alpha = (kc/(2*rc))**0.5
  ndim = len(axes)
  # direct-space part is independent of spatial dimensions
  def vsr_of_r(r):
    return erfc(alpha*r)/r
  vlr_r0 = 2*alpha/np.pi**0.5
  # reciprocal-space part changes with dimension
  if ndim == 2:
    vsr_k0 = 2*np.pi**0.5/alpha
    def vlr_of_k(k):
      return 2*np.pi/k*erfc(k/(2*alpha))
  elif ndim == 3:
    vsr_k0 = np.pi/(alpha*alpha)
    def vlr_of_k(k):
      k2 = k*k
      return 4*np.pi/k2*np.exp(-k2/(4*alpha*alpha))
  else:
    msg = 'unknown ndim = %d' % ndim
    raise RuntimeError(msg)
  # perform Ewald sums
  bmat = raxes(amat)
  vsr = sum_lattice(vsr_of_r, amat, rc)
  vlr = sum_lattice(vlr_of_k, bmat, kc)
  vol = volume(amat)
  vsr = (vsr-vlr_r0)/2
  vlr = (vlr-vsr_k0)/vol/2
  vmad = vsr+vlr
  return vmad

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
    assert len(nx) == ndim
    nxnynz = [np.arange(n) for n in nx]
  except TypeError:
    nxnynz = [np.arange(nx)]*ndim
  pos = np.stack(
    np.meshgrid(*nxnynz, indexing='ij'),
    axis=-1
  ).reshape(-1, ndim)
  return pos

def pos_in_bz(kvecs, raxes, nsh=3):
  """ put kvectors into the first Brillouin zone

  Args:
    kvecs (np.array): shape (nkpt, ndim), kpoints
    raxes (np.array): shape (ndim, ndim), reciprocal cell
  Return:
    np.array: shape (nkpt, ndim), kpoints in BZ
  """
  ndim = len(raxes)
  # create 1 shell of reciprocal lattice
  gvecs = np.dot(cubic_pos(nsh, ndim=ndim)-nsh//2, raxes)
  # assign kpoints to rec. latt.
  drkg = kvecs[np.newaxis]-gvecs[:, np.newaxis]
  rkg = np.linalg.norm(drkg, axis=-1)
  idx = np.argmin(rkg, axis=0)
  # move to first BZ
  return kvecs - gvecs[idx]

def get_nvecs(axes, pos, atol=1e-10):
  """ find integer vectors of lattice positions from unit cell

  Args:
    axes (np.array): lattice vectors in row-major
    pos (np.array): lattice sites
  Return:
    np.array: nvecs, integer vectors that label the lattice sites
  Example:
    >>> nvecs = get_nvecs(axes, pos)
  """
  ncands = np.dot(pos, np.linalg.inv(axes))  # candidates
  nvecs = np.around(ncands).astype(int)  # convert to integer
  success = np.allclose(np.dot(nvecs, axes), pos, atol=atol)  # check
  if not success:
    raise RuntimeError('problem in get_nvecs')
  return nvecs

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
    disp = displacement(axes, pos[i], pos[j])
    dist  = np.linalg.norm(disp)
    dtable[i, j] = dtable[j, i] = dist
  return dtable

def minimum_image_displacements(axes, pos, rj=None, mnx=-1, mxx=1):
  """Calculate minimum-image displacement vectors between two sets of particles

  Args:
    axes (np.array): lattice vectors in row-major
    pos  (np.array): particle positions in row-major
  Return:
    tuple: (displacements, distances) shapes are
      (ni, nj, ndim) and (ni, nj), respectively
  """
  from itertools import product
  ri = pos
  if rj is None:
    rj = ri
  ni = ri.shape[0]; ndim = ri.shape[-1]
  assert rj.shape[-1] == ndim
  nj = len(rj)
  disps = np.zeros([ni, nj, ndim])
  dists = np.inf*np.ones([ni, nj])
  for l in range(ndim):
    for g in product(range(mnx, mxx+1), repeat=ndim):
      a = np.dot(g, axes)
      rj1 = rj+a
      drij = ri[:, np.newaxis] - rj1[np.newaxis, :]
      rij = np.linalg.norm(drij, axis=-1)
      sel = rij < dists
      dists[sel] = rij[sel]
      disps[sel] = drij[sel]
  return disps, dists

def displacement_table(axes, pos1, pos0):
  """Calculate the distance table between two sets of particles

  Args:
    axes (np.array): lattice vectors in row-major
    pos1 (np.array): particle positions in row-major
    pos0 (np.array): reference particle positions in row-major
  Return:
    np.array: dtable shape (npart1, npart0), entry [i, j] is pos1[i]-pos0[j]
  """
  drij = pos1[:, np.newaxis] - pos0[np.newaxis]
  # apply PBC
  box = np.diag(axes)
  if not np.allclose(np.diag(box), axes):
    drij, rij = minimum_image_displacements(axes, pos1, pos0)
  else:
    nint = np.around(drij)/box
    drij -= nint*box
  return drij

def find_dimers(rij, rmax=np.inf, rmin=0, sort_id=False):
  """ find all dimers within a separtion of (rmin, rmax)

  Args:
    rij  (np.array): distance table
    rmax (float, optional): maximum dimer separation, default np.inf
    rmin (float, optional): minimum dimer separation, default 0
    sort_id (bool, optional): sort pair by first atom id, default false
  Return:
    np.array: unique pairs, a list of (int, int) particle id pairs
  """
  natom, natom1 = rij.shape
  assert natom1 == natom
  found = np.zeros(natom, dtype=bool)
  pairs = []
  # loop through pair distances from small to large
  idx = np.triu_indices(natom, 1)
  dists = rij[idx]
  ij = np.array(idx).T
  for idist in np.argsort(dists):
    i, j = ij[idist]  # pair indices
    if found[i] or found[j]:
      continue
    rb = dists[idist]  # bond length
    if (rb < rmin) or (rb > rmax):
      continue
    pair = (i, j) if i < j else (j, i)
    pairs.append(pair)
    found[i] = True
    found[j] = True
    if np.all(found):
      break
  pa = np.array(pairs)
  if sort_id:  # sort pairs by first atom id
    i1 = np.argsort(pa[:, 0])
    sorted_pairs = pa[i1]
  else:
    sorted_pairs = pa
  return sorted_pairs

def dimer_rep(atoms, rmax=np.inf, rmin=0.0,
  sort_id=False, return_pairs=False):
  """Find dimer representation of atoms

  Args:
    atoms (ase.Atoms): one-component system
    rmax (float, optional): maximum dimer separation, default np.inf
    return_pairs (bool, optional): return indices of dimers, default False
  Return:
    (np.array, np.array): (com, avecs), center of mass and
     half-bond vector, one for each dimer
  """
  assert len(np.unique(atoms.get_chemical_symbols())) == 1
  pos = atoms.get_positions()
  # check orthorhombic
  axes = atoms.get_cell()
  box = np.diag(axes)
  is_orthorhombic = np.allclose(np.diag(box), axes)
  use_fortran = False
  if is_orthorhombic:  # try use FORTRAN routine for displacements
    try:
      from qharv.inspect.forlib.pbcbox import pbcbox
      use_fortran = True
    except ImportError as err:
      msg = 'please compile pbcbox using f2py for fast implementation'
      print(msg)
  if use_fortran:
    drij = -pbcbox.displacement_table(pos, box)
  else:
    drij = atoms.get_all_distances(mic=True, vector=True)
  rij = np.linalg.norm(drij, axis=-1)
  pairs = find_dimers(rij, rmax=rmax, rmin=rmin, sort_id=sort_id)
  # a vector points from particle 0 towards 1
  avecs = 0.5*drij[pairs[:, 0], pairs[:, 1]]
  com = pos[pairs[:, 0]] + avecs
  if return_pairs:
    ret = (com, avecs, pairs)
  else:
    ret = (com, avecs)
  return ret

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

# ======================== level 2: wrapping ========================
def linecut(axes, r0, dr, mr=10000, sort=True, fraction=True):
  """ generate a line across the cell

  Args:
    axes (np.array): lattice vectors, shape (ndim, ndim)
    r0 (np.array): starting position, shape (ndim,)
    dr (np.array): line direction, shape (ndim,)
    mr (int, optional): maximum number of points along line, default 10000
    sort (bool, optional): sort points along line, default True
    fraction (bool, optional): r0 and dr are in fractional coordinates
  Return:
    np.array: a list of points on the line, shape (npt, ndim)
  """
  if fraction:
    r0 = np.dot(r0, axes)
    dr = np.dot(dr, axes)
  ainv = np.linalg.inv(axes)
  line = []
  iline = []
  for pm in [1, -1]:
    for ir in range(mr):
      if (ir == 0) and (pm == -1): continue
      r = r0+pm*ir*dr
      f = np.dot(r, ainv)
      if np.any(f>1) or np.any(f<0):
        break
      line.append(r)
      iline.append(ir*pm)
    if ir >= mr-1:
      msg = 'not enough points to reach edge of cell'
      msg += ' increase dr=%f or mr=%d' % (dr, mr)
      raise RuntimeError(msg)
  rline = np.array(line)
  if sort:
    idx = np.argsort(iline)
    rline = rline[idx]
  return rline

# ==================== level 2: space partitions ====================

def rcut_partition(axes, pos, rvecs, rcut=None):
  if rcut is None:
    disps, dists = minimum_image_displacements(axes, pos)
    natom = len(pos)
    idx = np.triu_indices(natom, k=1)
    dist_min = dists[idx].min()
    rcut = dist_min/2
  factlist = np.zeros(len(rvecs), dtype=int)
  for i, p in enumerate(pos):
    drij, rij = minimum_image_displacements(axes, p[np.newaxis], rvecs)
    sel = rij[0] < rcut
    factlist[sel] = i+1
  return factlist

def voronoi_partition(axes, pos, rvecs):
  factlist = np.zeros(len(rvecs), dtype=int)
  for i, r1 in enumerate(rvecs):
    disps, dists = minimum_image_displacements(axes, r1[np.newaxis], pos)
    factlist[i] = np.argmin(dists[0])+1
  return factlist
