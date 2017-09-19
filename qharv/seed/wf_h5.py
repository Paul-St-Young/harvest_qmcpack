# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to read the QMCPACK wavefunction hdf5 file, usually named pwscf.pwscf.h5
import os
import h5py
import numpy as np

def read(fname):
  return h5py.File(fname)

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

dtypes = {
  'gvectors':int,
  'nkpt':int,
  'nspin':int,
  'nstate':int,
}

def get(name,fp):
  if name not in locations.keys():
    raise RuntimeError('unknown attribute requested: %s' % name)
  return fp[ locations[name] ].value
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

def get_orb_in_pw(ikpt,ispin,istate,fp):
  orb_path = os.path.join( state_path(ikpt,ispin,istate), 'psi_g' )
  psig_arr = fp[orb_path].value # stored in real view
  #psig = psig_arr[:,0]+1j*psig_arr[:,1] # convert to complex view
  psig = psig_arr.flatten().view(complex) # more elegant conversion
  return psig
# =======================================================================

