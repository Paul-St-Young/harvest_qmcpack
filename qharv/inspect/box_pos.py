# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to cubic structure specified by pos and box
import numpy as np

def pos_in_box(pos, box):
  """ Enforce periodic bounary condition (PBC) on particle positions.
  Simulation box contains [0, Lx)x[0, Ly)x[0, Lz)

  Args:
    pos (np.array): particle positions under open boundary conditions.
    box (float): side lengths of box
  Return:
    np.array: particle positions under periodic bounary conditions.
  """
  ndim = pos.shape[-1]
  try:
    ndim = len(box)  # except if box is float
  except TypeError as err:  # take care of simple case (lx=ly=lz)
    return pos % box
  assert len(box) == ndim
  pos1 = pos.copy()
  for idim in range(ndim):
    pos1[:, idim] = pos1[:, idim] % box[idim]
  return pos1

def disp_in_box(drij, box):
  """ Enforce minimum image convention (MIC) on displacement vectors.

  Args:
    drij (np.array): ri - rj for all pairs of particles rij
    box (float): side lengths of box
  Return:
    np.array: displacement vectors under MIC.
  """
  ndim = drij.shape[-1]
  try:
    ndim = len(box)  # except if box is float
  except TypeError as err:  # take care of simple case (lx=ly=lz)
    nint = np.around(drij/box)
    return drij-box*nint
  assert len(box) == ndim
  drij1 = drij.copy()
  for idim in range(ndim):
    nint = np.around(drij[:, :, idim]/box[idim])
    drij1[:, :, idim] -= box[idim]*nint
  return drij1

def displacement_table(pos, box):
  """ Calculate displacements ri-rj between all pairs of particles.

  Args:
    pos (np.array): particle positions
    box (float): side lengths of box
  Return:
    np.array: drij, a table of displacement vectors under MIC,
     shape (natom, natom, ndim)
  """
  drij = pos[:, np.newaxis] - pos[np.newaxis]
  return disp_in_box(drij, box)
