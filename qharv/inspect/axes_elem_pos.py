# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to process crystal structure specified by axes, elem, pos
#  extend axes_pos to handle more than 1 species of atoms
import numpy as np

def ase_read(floc):
  """ use atomic simulation environment (ase) package to read file
  can read any format ase.read supports

  Args:
    floc (str): file location of structure file (e.g. struct.xsf)
  Return:
    dict: with ['axes', 'elem', 'pos'] entries
  """
  from ase.io import read

  s1 = read(floc)
  axes = s1.get_cell().tolist()
  elem = s1.get_chemical_symbols()
  pos = s1.get_positions().tolist()
  data = {
    'axes': axes,
    'elem': elem,
    'pos': pos
  }
  return data

def ase_tile(axes, elem, pos, tmat):
  """ use ase to tile supercell

  Args:
    axes (np.array): lattice vectors in row-major
    elem (np.array): chemical symbols
    pos (np.array): atomic positions
    tmat (np.array): tiling matrix (a.k.a. supercell matrix)
  Return:
    (np.array, np.array, np.array): supercell ('axes', 'elem', 'pos')
  """
  from ase import Atoms
  from ase.build import make_supercell
  pbc = [True]*3  # assume PBC if tiling
  s0 = Atoms(''.join(elem), cell=axes, positions=pos, pbc=pbc)
  s1 = make_supercell(s0, tmat)
  axes1 = s1.get_cell()
  elem1 = s1.get_chemical_symbols()
  pos1 = s1.get_positions()
  return axes1, elem1, pos1
