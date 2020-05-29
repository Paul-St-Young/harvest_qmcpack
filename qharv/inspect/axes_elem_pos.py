# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to process crystal structure specified by axes, elem, pos
#  extend axes_pos to handle more than 1 species of atoms
#  heavily borrow from the atomic simulation environment (ase) package
import numpy as np

# ======================== level 0: construct atoms =========================

def default_pbc(axes, kwargs):
  pbc = kwargs.pop('pbc', None)
  if pbc is None:
    ndim = len(axes)
    pbc = [True]*ndim
  return pbc

def ase_atoms(axes, elem, pos, **kwargs):
  """ create ase Atoms object

  Args:
    axes (np.array): lattice vectors in row-major
    elem (np.array): chemical symbols
    pos (np.array): atomic positions
  Return:
    ase.Atoms: Atoms object
  """
  from ase import Atoms
  pbc = default_pbc(axes, kwargs)
  s0 = Atoms(''.join(elem), cell=axes, positions=pos, pbc=pbc, **kwargs)
  return s0

def make_atoms(axes, posl, eleml=None, **kwargs):
  """ create ase Atoms object using arbitrary element names

  Args:
    axes (np.array): lattice vectors in row-major
    posl (list): a list of atomic positions, each a np.array
    eleml (list, optional): a list of chemical symbols
  Return:
    ase.Atoms: Atoms object
  """
  from ase import Atoms
  pbc = default_pbc(axes, kwargs)
  if eleml is None:
    eleml = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
  pos = []
  elem = []
  for e1, p1 in zip(eleml, posl):
    elem += [e1]*len(p1)
    pos += p1.tolist()
  s0 = Atoms(''.join(elem), cell=axes, positions=pos, pbc=pbc, **kwargs)
  return s0

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

# ======================== level 0: use methods =========================

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

def ase_drij(atoms, mic=True):
  if mic:
    assert np.allclose(atoms.get_pbc(), 1)
  return atoms.get_all_distances(mic=mic, vector=True)

def ase_rij(atoms, mic=True):
  if mic:
    assert np.allclose(atoms.get_pbc(), 1)
  return atoms.get_all_distances(mic=mic)
