# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to cubic structure specified by pos and lbox
import numpy as np

def pos_in_box(pos, lbox):
  """ Enforce periodic bounary condition (PBC) on particle positions.
  Simulation box contains [0, L)

  Args:
    pos (np.array): particle positions under open boundary conditions.
    lbox (float): side length of cubic box
  Return:
    np.array: particle positions under periodic bounary conditions.
  """
  return pos % lbox

def disp_in_box(drij, lbox):
  """ Enforce minimum image convention (MIC) on displacement vectors.

  Args:
    drij (np.array): ri - rj for all pairs of particles rij
    lbox (float): side length of cubic box
  Return:
    np.array: displacement vectors under MIC.
  """
  nint = np.around(drij/lbox)
  return drij-lbox*nint

def displacement_table(pos, lbox):
  """ Calculate displacements ri-rj between all pairs of particles.

  Args:
    pos (np.array): particle positions
    lbox (float): side length of cubic box
  Return:
    np.array: drij, a table of displacement vectors under MIC,
     shape (natom, natom, ndim)
  """
  drij = pos[:, np.newaxis] - pos[np.newaxis]
  return disp_in_box(drij, lbox)
