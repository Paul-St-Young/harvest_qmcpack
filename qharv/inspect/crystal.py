# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to extract and visualize crystal structure data

import numpy as np
from qharv.seed import xml

def lattice_vectors(fname):
  doc  = xml.read(fname)
  axes = xml.get_axes(doc)
  return axes

def atomic_coords(fname,pset='ion0'):
  doc = xml.read(fname)
  pos = xml.get_pos(doc,pset=pset)
  return pos

def draw_atoms(ax,pos,**kwargs):
  """ draw atoms on ax
  see example in draw_crystal

  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   pos (np.array): array of atomic positions
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D
  """
  # set defaults
  if not ( ('c' in kwargs) or ('color' in kwargs) ):
    kwargs['c'] = 'b'
  if not ( ('ls' in kwargs) or ('linestyle' in kwargs) ):
    kwargs['ls'] = ''
  if not ('marker' in kwargs):
    kwargs['marker'] = 'o'
  if not (('ms' in kwargs) or ('markersize' in kwargs)):
    kwargs['ms'] = 10
  dots  = ax.plot(pos[:,0],pos[:,1],pos[:,2],**kwargs)
  return dots

def draw_cell(ax,axes,corner=None,enclose=True,**kwargs):
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
  cell = []
  if corner is None:
    corner = np.array([0,0,0])
  # end if

  # set defaults
  if not ( ('c' in kwargs) or ('color' in kwargs) ):
    kwargs['c'] = 'gray'
  if not ('alpha' in kwargs):
    kwargs['alpha'] = 0.6
  if not ( ('lw' in kwargs) or ('linewidth' in kwargs) ):
    kwargs['lw'] = 2

  # a,b,c lattice vectors
  for iax in range(3):
    start = corner
    end   = start + axes[iax]
    line = ax.plot(*zip(start,end),**kwargs)
    cell.append(line)
  # end for iax

  if enclose:
    # counter a,b,c vectors
    for iax in range(3):
      start = corner+axes.sum(axis=0)
      end   = start - axes[iax]
      line = ax.plot(*zip(start,end),**kwargs)
      cell.append(line)
    # end for iax
    
    # remaining vectors needed to enclose cell
    for iax in range(3):
      start = corner+axes[iax]
      for jax in range(3):
        if jax == iax:
          continue
        end = start + axes[jax]
        line = ax.plot(*zip(start,end),**kwargs)
        cell.append(line)
      # end for jax
    # end for iax
  # end if

  return cell

def draw_crystal(ax,axes,pos,draw_super=False):
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
   list,list: (atoms,cell) a list of plt.Line3D for the atoms,
    a list of plt.Line3D for the cell.
  """
  # draw primitive cell
  cell = draw_cell(ax,axes)
  dots = draw_atoms(ax,pos)
  atoms = [dots]

  nx = ny = nz = 2 # !!!! hard-code 2x2x2 supercell
  if draw_super: # draw supercell
    import numpy as np
    from itertools import product
    for ix,iy,iz in product(range(nx),range(ny),range(nz)):
      if ix==iy==iz==0:
        continue
      # end if
      #shift = (np.array([ix,iy,iz])*axes).sum(axis=0)
      shift = ix*axes[0] + iy*axes[1] + iz*axes[2]
      spos  = (shift.reshape(-1,1,3) + pos).reshape(-1,3)
      dots  = draw_atoms(ax,spos,ls='',marker='o',c='gray',ms=10,alpha=0.8)
      atoms.append(dots)
    # end for
  # end if

  return atoms,cell
# end def
