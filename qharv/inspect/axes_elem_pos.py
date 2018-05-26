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
