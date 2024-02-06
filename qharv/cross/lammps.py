import numpy as np

def dump_txt(traj):
  # check implementation
  atoms = traj[0]
  axes = atoms.get_cell()
  box = np.diag(axes)
  if not np.allclose(np.diag(box), axes):
    msg = 'non-orthogonal box'
    raise NotImplementedError(msg)
  elem = atoms.get_chemical_symbols()
  if len(np.unique(elem)) != 1:
    raise NotImplementedError(str(elem))
  # write dump
  text = ''
  for i, atoms in enumerate(traj):
    # META
    text += 'ITEM: TIMESTEP\n'
    text += '%d\n' % (i+1)
    #
    text += 'ITEM: NUMBER OF ATOMS\n'
    text += '%d\n' % len(atoms)
    # BOX
    text += 'ITEM: BOX BOUNDS pp pp pp\n'
    box = np.diag(atoms.get_cell())
    for b in box:
      text += '0.0 %f\n' % b
    # ATOMS
    pos = atoms.get_positions()
    text += 'ITEM: ATOMS id type x y z\n'
    for i, p in enumerate(pos):
      text += '%d 1 %f %f %f\n' % (i+1, *p)
  return text
