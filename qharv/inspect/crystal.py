# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to extract and visualize crystal structure data

import numpy as np
from qharv.seed import xml

def lattice_vectors(fname):
  """ extract lattice vectors from QMCPACK input
  similar to ase.Atoms.get_cell()

  Args:
    fname (str): xml input filename
  Return:
    np.array: axes
  """
  doc  = xml.read(fname)
  axes = xml.get_axes(doc)
  return axes

def atomic_coords(fname, pset='ion0'):
  """ extract atomic positions from QMCPACK input
  similar to ase.Atoms.get_positions()

  Args:
    fname (str): xml input filename
  Return:
    np.array: axes
  """
  doc = xml.read(fname)
  pos = xml.get_pos(doc, pset=pset)
  return pos

def set_default_atoms_styles(kwargs):
  if not (('c' in kwargs) or ('color' in kwargs)):
    kwargs['c'] = 'b'
  if not ('alpha' in kwargs):
    kwargs['alpha'] = 0.25
  if not (('ls' in kwargs) or ('linestyle' in kwargs)):
    kwargs['ls'] = ''
  if not ('marker' in kwargs):
    kwargs['marker'] = 'o'
  if not (('ms' in kwargs) or ('markersize' in kwargs)):
    kwargs['ms'] = 5

def draw_atoms(ax, pos, **kwargs):
  """ draw atoms on ax
  see example in draw_crystal

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   pos (np.array): array of atomic positions
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D
  """
  set_default_atoms_styles(kwargs)
  dots  = ax.plot(*pos.T, **kwargs)
  return dots

def draw_dimers(ax, com, bonds, **kwargs):
  """ draw dimers on ax

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   com (np.array): center of mass of dimers
   bonds (np.array): bond vectors of dimers
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   tuple: 3 plot objects (monomer1, monomer2, bond)
  """
  set_default_atoms_styles(kwargs)
  r1 = com - 0.5*bonds
  kwargs['c'] = 'b'
  dots1  = ax.plot(*r1.T, **kwargs)
  r2 = com + 0.5*bonds
  kwargs['c'] = 'r'
  dots2  = ax.plot(*r2.T, **kwargs)
  x, y, z = r1.T
  vx, vy, vz = bonds.T
  qv = ax.quiver(x, y, z, vx, vy, vz)
  return dots1, dots2, qv

def draw_forces(ax, pos, vel, **kwargs):
  """ draw forces on atoms

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   pos (np.array): array of atomic positions
   vel (np.array): array of forces on each atom (or velocities)
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D
  """
  x, y, z = zip(*pos)
  vx, vy, vz = zip(*vel)
  qvs = ax.quiver(x, y, z, vx, vy, vz, **kwargs)
  return qvs

def set_default_cell_styles(kwargs):
  if not (('c' in kwargs) or ('color' in kwargs)):
    kwargs['c'] = 'gray'
  if not ('alpha' in kwargs):
    kwargs['alpha'] = 0.6
  if not (('lw' in kwargs) or ('linewidth' in kwargs)):
    kwargs['lw'] = 2

def draw_cell(ax, axes, corner=None, enclose=True, **kwargs):
  """ draw cell on ax
  see example in draw_crystal

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   axes (np.array): lattice vectors in row-major 3x3 array
   corner (np.array,optional): lower left corner of the lattice
     ,use (0,0,0) by default
   enclose (bool): enclose the cell with lattice vectors
     ,default is True. If False, then draw lattice vectors only
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D, one for each lattice vector
  """
  ndim = len(axes)
  if ndim not in [2, 3]:
    raise RuntimeError('ndim = %d is not supported' % ndim)
  cell = []
  if corner is None:
    corner = np.zeros(ndim)

  set_default_cell_styles(kwargs)

  # a,b,c lattice vectors
  for iax in range(ndim):
    start = corner
    end   = start + axes[iax]
    line = ax.plot(*zip(start, end), **kwargs)
    cell.append(line)

  if enclose:
    # counter a,b,c vectors
    for iax in range(ndim):
      start = corner+axes.sum(axis=0)
      end   = start - axes[iax]
      line = ax.plot(*zip(start, end), **kwargs)
      cell.append(line)

    if ndim > 2:
      # remaining vectors needed to enclose cell
      for iax in range(ndim):
        start = corner+axes[iax]
        for jax in range(ndim):
          if jax == iax:
            continue
          end = start + axes[jax]
          line = ax.plot(*zip(start, end), **kwargs)
          cell.append(line)
  # end if enclose
  return cell

def draw_wigner_seitz_cell(ax, axes, nsh=1, **kwargs):
  from scipy.spatial import Voronoi
  set_default_cell_styles(kwargs)
  # create Voronoi tessellation
  from qharv.inspect.axes_pos import cubic_pos
  qvecs = cubic_pos(2*nsh+1, ndim=len(axes))-nsh
  dots = np.dot(qvecs, axes)
  vor = Voronoi(dots)
  verts = vor.vertices  # vertices (basis for the rest)
  regs = vor.regions
  rverts = vor.ridge_vertices  # ridges separate regions
  # find vertices of THE enclosed region
  enclosed_regions = []
  for reg in regs:
    if len(reg) == 0:
      continue
    if -1 in reg:
      continue
    enclosed_regions.append(reg)
  nreg = len(enclosed_regions)
  if not nreg == 1:
    raise RuntimeError('found %d region; try increase nsh' % nreg)
  ereg = enclosed_regions[0]
  # find rigdes that enclose this region
  #  also append first region to close each face
  myrverts = []
  for rvert in rverts:
    # skip all with unknown vertices
    if -1 in rvert:
      continue
    # skip all not belonginig to THE region
    skip = False
    for iv in rvert:
      if iv not in ereg:
        skip = True
    if skip:
      continue
    myrvert = rvert+[rvert[0]]  # close face
    myrverts.append(myrvert)
  # draw enclosing regions
  lines = []
  for rvert in myrverts:
    if -1 in rvert:
      continue
    pts = verts[rvert]
    line = ax.plot(*pts.T, **kwargs)
    lines.append(line)
  return lines

# ======================== composition =========================
def draw_crystal(ax, axes, pos, draw_super=False):
  """ draw crystal structure on ax

  Example:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    axes = np.eye(3)
    pos  = np.array([ [0.5,0.5,0.5] ])

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1,projection='3d')
    draw_crystal(ax,axes,pos)
    plt.show()

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   axes (np.array): lattice vectors in row-major 3x3 array
   pos (np.array): array of atomic positions
   draw_super (bool): draw 2x2x2 supercell
  Returns:
   list,list: (cell, atoms) cell is a list of plt.Line3D for the cell,
   atoms is a list of plt.Line3D for the atoms.
  """
  # draw primitive cell
  cell = draw_cell(ax, axes)
  dots = draw_atoms(ax, pos)
  atoms = [dots]

  if draw_super:  # draw supercell
    nx = ny = nz = 2  # !!!! hard-code 2x2x2 supercell
    import numpy as np
    from itertools import product
    for ix, iy, iz in product(range(nx), range(ny), range(nz)):
      if ix == iy == iz == 0:
        continue
      # end if
      #shift = (np.array([ix,iy,iz])*axes).sum(axis=0)
      shift = ix*axes[0] + iy*axes[1] + iz*axes[2]
      spos  = (shift.reshape(-1, 1, 3) + pos).reshape(-1, 3)
      dots  = draw_atoms(ax, spos)
      atoms.append(dots)
    # end for
  # end if
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  return cell, atoms
