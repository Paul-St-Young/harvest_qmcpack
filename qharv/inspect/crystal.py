# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to extract and visualize crystal structure data

import numpy as np
from qharv.seed import xml

def lattice_vectors(fname):
  doc = xml.read(fname)
  sc_node = doc.find('.//simulationcell')
  if sc_node is None:
    raise RuntimeError('<simulationcell> not found in %s'%fname)
  lat_node = sc_node.find('.//parameter[@name="lattice"]')
  unit = lat_node.get('units')
  assert unit == 'bohr'
  axes = xml.text2arr( lat_node.text )
  return axes

def atomic_coords(fname,pset_name='ion0'):
  # !!!! assuming atomic units (bohr)
  # !!!! finds the first group in particleset
  doc = xml.read(fname)
  source_pset_node = doc.find('.//particleset[@name="%s"]'%pset_name)
  if source_pset_node is None:
    raise RuntimeError('%s not found in %s'%(pset_name,fname))
  pos_node = source_pset_node.find('.//attrib[@name="position"]')
  pos = xml.text2arr(pos_node.text)
  return pos

def draw_atoms(ax,pos,**kwargs):
  """ draw atoms on ax
  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   pos (np.array): array of atomic positions
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D
  """
  dots  = ax.plot(pos[:,0],pos[:,1],pos[:,2],**kwargs)
  return dots

def draw_cell(ax,axes,**kwargs):
  """ draw cell on ax
  Args:
   ax (plt.Axes): matplotlib Axes object, must have projection='3d'
   axes (np.array): lattice vectors in row-major 3x3 array
   kwargs (dict,optional): keyword arguments passed to plt.plot
  Returns:
   list: a list of plt.Line3D, one for each lattice vector
  """
  cell = []

  # set defaults
  if not ( ('c' in kwargs) or ('color' in kwargs) ):
    kwargs['c'] = 'gray'
  if not ('alpha' in kwargs):
    kwargs['alpha'] = 0.6
  if not ( ('lw' in kwargs) or ('linewidth' in kwargs) ):
    kwargs['lw'] = 2

  # a,b,c lattice vectors
  for iax in range(3):
    start = np.array([0,0,0])
    end   = start + axes[iax]
    line = ax.plot(*zip(start,end),**kwargs)
    cell.append(line)
  # end for iax

  # counter a,b,c vectors
  for iax in range(3):
    start = axes.sum(axis=0)
    end   = start - axes[iax]
    line = ax.plot(*zip(start,end),**kwargs)
    cell.append(line)
  # end for iax
  
  # remaining vectors needed to enclose cell
  for iax in range(3):
    start = axes[iax]
    for jax in range(3):
      if jax == iax:
        continue
      end = start + axes[jax]
      line = ax.plot(*zip(start,end),**kwargs)
      cell.append(line)
    # end for jax
  # end for iax

  return cell

def draw_crystal(ax,axes,pos,draw_super=False):
  """ draw crystal structure on ax
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
  dots = draw_atoms(ax,pos,ls='',marker='o',c='b',ms=10)
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
