# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to extract and visualize crystal structure data

import numpy as np
from qharv.seed import xml

def lattice_vectors(fname):
  doc = xml.read(fname)
  sc_node = doc.find('.//simulationcell')
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
  pos_node = source_pset_node.find('.//attrib[@name="position"]')
  pos = xml.text2arr(pos_node.text)
  return pos

def draw_cell(ax,axes,pos,atom_color='b',draw_super=False):
  atoms = []
  dots  = ax.plot(pos[:,0],pos[:,1],pos[:,2],'o',c=atom_color,ms=10)
  atoms.append(dots)

  nx = ny = nz = 2 # !!!! hard-code 2x2x2 supercell
  if draw_super: # draw supercell
    import numpy as np
    from itertools import product
    for ix,iy,iz in product(range(nx),range(ny),range(nz)):
      if ix==iy==iz==0:
        continue
      #shift = (np.array([ix,iy,iz])*axes).sum(axis=0)
      shift = ix*axes[0] + iy*axes[1] + iz*axes[2]
      spos  = (shift.reshape(-1,1,3) + pos).reshape(-1,3)
      dots  = ax.plot(spos[:,0],spos[:,1],spos[:,2],'o',c='gray',ms=10,alpha=0.8)
      atoms.append(dots)

  # show primitive cell
  cell = []
  for idim in range(3):
    line = ax.plot([0,axes[idim,0]],[0,axes[idim,1]],[0,axes[idim,2]],c='k',lw=2)
    cell.append(line)

  return atoms,cell
# end def
