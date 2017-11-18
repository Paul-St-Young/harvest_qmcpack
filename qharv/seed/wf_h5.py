# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to read the QMCPACK wavefunction hdf5 file, usually named pwscf.pwscf.h5
import os
import h5py
import numpy as np

def read(fname,mode='r'):
  return h5py.File(fname,mode)

# =======================================================================
# QMCPACK wavefunction hdf5 fixed locations
# =======================================================================
locations = {
  'gvectors':'electrons/kpoint_0/gvectors',
  'nkpt':'electrons/number_of_kpoints',
  'nspin':'electrons/number_of_spins',
  'nstate':'electrons/kpoint_0/spin_0/number_of_states', # !!!! same number of states per kpt
  'axes':'supercell/primitive_vectors',
  'pos':'atoms/positions'
}

def get(fp,name):
  if name not in locations.keys():
    raise RuntimeError('unknown attribute requested: %s' % name)
  return fp[ locations[name] ].value

def axes_elem_pos(fp):
  """ extract lattice vectors, atomic positions, and element names 
  from wavefunction hdf5 file
  Args:
    fp (h5py.File): hdf5 file pointer
  Returns:
    (np.array,list[str],np.array): (axes,elem,pos)
  """
  axes = fp[ locations['axes'] ].value
  pos  = fp[ locations['pos'] ].value

  # construct list of atomic labels
  elem_id  = fp['atoms/species_ids'].value
  elem_map = {}
  nelem = fp['atoms/number_of_species'].value
  for ielem in range(nelem):
    elem_name = fp['atoms/species_%d/name'%ielem].value[0]
    elem_map[ielem] = elem_name
  # end for ielem
  elem = [elem_map[eid] for eid in elem_id]
  assert len(elem) == len(pos)
  return axes,elem,pos

# =======================================================================

# =======================================================================
# QMCPACK wavefunction hdf5 orbital locations
# =======================================================================
def kpoint_path(ikpt):
  """ construct path to kpoint

  e.g. electrons/kpoint_0/spin_0/state_0

  Args: 
   ikpt (int): kpoint index
  Returns:
   str: path in hdf5 file
  """
  path = 'electrons/kpoint_%d' % (ikpt)
  return path
def spin_path(ikpt,ispin):
  path = 'electrons/kpoint_%d/spin_%d' % (ikpt,ispin)
  return path
def state_path(ikpt,ispin,istate):
  path = 'electrons/kpoint_%d/spin_%d/state_%d/' % (ikpt,ispin,istate)
  return path

def get_orb_in_pw(fp,ikpt,ispin,istate):
  orb_path = os.path.join( state_path(ikpt,ispin,istate), 'psi_g' )
  psig_arr = fp[orb_path].value # stored in real view
  #psig = psig_arr[:,0]+1j*psig_arr[:,1] # convert to complex view
  psig = psig_arr.flatten().view(complex) # more elegant conversion
  return psig
# =======================================================================

