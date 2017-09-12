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

def xsf_datagrid_3d_density(fname):
  from qharv.reel import ascii_out
  mm   = ascii_out.read(fname)
  text = ascii_out.block_text(mm,'BEGIN_DATAGRID_3D_density','END_DATAGRID_3D_density')

  lines = text.split('\n')

  # first advance iline past particle coordinates
  iline = 0
  for line in lines:
    if iline == 0:
      grid_shape = map(int,lines[0].split())
      iline += 1
      continue
    # end if

    tokens = line.split()
    if len(tokens) == 3: # atom coordinate
      pass
    elif len(tokens) == 4: # density data
      break
    # end if
    iline += 1
  # end for line

  # then convert data to density grid
  data = np.array( [text.split() for text in lines[iline:-1]],dtype=float)
  return data.reshape(grid_shape)
# end def xsf_datagrid_3d_density

def draw_cell(ax,axes,pos,atom_color='b'):
  atoms = ax.plot(pos[:,0],pos[:,1],pos[:,2],'o',c=atom_color,ms=10)

  # show simulation cell
  cell = []
  for idim in range(3):
    line = ax.plot([0,axes[idim,0]],[0,axes[idim,1]],[0,axes[idim,2]],c='k',lw=2)
    cell.append(line)
  return atoms,cell
# end def
