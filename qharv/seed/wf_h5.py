# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to read the QMCPACK wavefunction hdf5 file
#  Mostly built around h5py module's API.
#  The central object is h5py.File, which is usually named "fp".
import os
import h5py
import numpy as np

# ====================== level 0: basic io =======================


def read(fname, **kwargs):
  """ read h5 file and return a h5py File object

  Args:
    fname (str): hdf5 file
    kwargs (dict): keyword arguments to pass on to h5py.File,
      default is {'mode':'r'}
  Return:
    h5py.File: h5py File object
  """
  if not ('mode' in kwargs):
    kwargs['mode'] = 'r'
  return h5py.File(fname, **kwargs)


def ls(handle, r=False, level=0, indent="  "):
  """ List directory structure

   Similar to the Linux `ls` command, but for an hdf5 file

   Args:
     handle (h5py.Group): or h5py.File or h5py.Dataset
     r (bool): recursive list
     level (int): level of indentation, only used if r=True
     indent (str): indent string, only used if r=True
   Returns:
     str: mystr, a string representation of the directory structure
  """
  mystr = ''
  if isinstance(handle, h5py.File) or isinstance(handle, h5py.Group):
    for key, val in handle.items():
      mystr += indent*level+'/'+key + "\n"
      if r:
        mystr += ls(val, r=r, level=level+1, indent=indent)
    # end for
  elif isinstance(handle, h5py.Dataset):
    return ''
  else:
    raise RuntimeError('cannot handle type=%s' % type(handle))
  # end if
  return mystr


# ====== level 1: QMCPACK wavefunction hdf5 fixed locations ======
locations = {
  'gvectors': 'electrons/kpoint_0/gvectors',
  'nkpt': 'electrons/number_of_kpoints',
  'nelecs': 'electrons/number_of_electrons',
  'nspin': 'electrons/number_of_spins',
  'nstate': 'electrons/kpoint_0/spin_0/number_of_states',
  # !!!! assume same number of states per kpt
  'axes': 'supercell/primitive_vectors',
  'pos': 'atoms/positions'
}


def get(fp, name):
  """ retrieve data from a known location in pwscf.h5

  Args:
    fp (h5py.File): hdf5 file object
    name (str): a known name in locations
  Return:
    array_like: whatever fp[loc].value returns
  """
  if name not in locations.keys():
    msg = 'unknown attribute requested: "%s"' % name
    msg += '\n known attributes:\n  ' + '\n  '.join(locations.keys())
    raise RuntimeError(msg)
  loc = locations[name]
  return fp[loc].value


def axes_elem_pos(fp):
  """ extract lattice vectors, atomic positions, and element names
  The main difficulty is constructing the element names of each
  atomic species. If elem is not needed, use get(fp,'axes') and
  get(fp,'pos') to get the simulation cell and ion positions directly.

  Args:
    fp (h5py.File): hdf5 file object
  Returns:
    (np.array,list[str],np.array): (axes,elem,pos)
  """
  axes = get(fp, 'axes')
  pos  = get(fp, 'pos')

  # construct list of atomic labels
  elem_id  = fp['atoms/species_ids'].value
  elem_map = {}
  nelem = fp['atoms/number_of_species'].value
  for ielem in range(nelem):
    elem_name = fp['atoms/species_%d/name' % ielem].value[0]
    elem_map[ielem] = elem_name
  # end for ielem
  elem = [elem_map[eid] for eid in elem_id]
  assert len(elem) == len(pos)
  return axes, elem, pos

# ====== level 2: QMCPACK wavefunction hdf5 orbital locations ======


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


def spin_path(ikpt, ispin):
  path = 'electrons/kpoint_%d/spin_%d' % (ikpt, ispin)
  return path


def state_path(ikpt, ispin, istate):
  path = 'electrons/kpoint_%d/spin_%d/state_%d/' % (ikpt, ispin, istate)
  return path


def get_orb_in_pw(fp, ikpt, ispin, istate):
  """ get the plane wave coefficients of a single Kohn-Sham orbital

  Args:
    fp (h5py.File): wf h5 file
    ikpt (int): kpoint index
    ispin (int): spin index
    istate (int): band index
  Return:
    (np.array, np.array): (gvecs, cmat), PWs and coefficient matrix
  """
  orb_path = os.path.join(state_path(ikpt, ispin, istate), 'psi_g')
  psig_arr = fp[orb_path].value  # stored in real view
  # psig = psig_arr[:,0]+1j*psig_arr[:,1]  # convert to complex view
  psig = psig_arr.flatten().view(complex)  # more elegant conversion
  return psig


# ====== level 3: single particle orbitals ======


def get_cmat(fp, ikpt, ispin):
  """ get Kohn-Sham orbital coefficients on a list of plane waves (PWs)

  Args:
    fp (h5py.File): wf h5 file
    ikpt (int): kpoint index
    ispin (int): spin index
  Returns:
    (np.array, np.array): (gvecs, cmat), PWs and coefficient matrix
  """
  # decide how many orbitals to extract (norb)
  nelecs = get(fp, 'nelecs')  # all spins
  norb = nelecs[ispin]  # get all occupied orbitals
  # count the number of PWs (npw)
  gvecs = get(fp, 'gvectors')  # integer vectors
  npw = len(gvecs)
  # construct coefficient matrix
  cmat = np.zeros([norb, npw], dtype=complex)
  for iorb in range(norb):
    ci = get_orb_in_pw(fp, ikpt, ispin, iorb)
    cmat[iorb, :] = ci
  return gvecs, cmat


def normalize_cmat(cmat):
  """ normalize PW orbital coefficients

  Args:
    cmat (np.array): coefficient matrix shape (norb, npw)
  Effect:
    each row of cmat will be normalized to |ci|^2=1
  """
  norb, npw = cmat.shape
  for iorb in range(norb):
    ci = cmat[iorb]
    norm = np.dot(ci.conj(), ci)
    cmat[iorb] /= norm**0.5


def get_twists(fp, ndim=3):
  """ return the list of available twist vectors

  Args:
    fp (h5py.File): wf h5 file
    ndim (int, optional): number of spatial dimensions, default 3
  Returns:
    np.array: tvecs, twist vectors in reciprocal lattice units (nk, ndim)
  """
  nk = get(fp, 'nkpt')[0]
  ukvecs = np.zeros([nk, ndim])
  for ik in range(nk):
    kpath = kpoint_path(ik)
    ukvec = fp[os.path.join(kpath, 'reduced_k')].value
    ukvecs[ik, :] = ukvec
  return ukvecs


def get_bands(fp, ispin=0):
  """ return the list of available Kohn-Sham eigenvalues

  Args:
    fp (h5py.File): wf h5 file
    ispin (int, optional): spin index, default 0
  Returns:
    np.array: tvecs, twist vectors in reciprocal lattice units (nk, nbnd)
  """
  nk = get(fp, 'nkpt')[0]
  nbnd = get(fp, 'nstate')[0]
  bands = np.zeros([nk, nbnd])
  for ik in range(nk):
    kpath = kpoint_path(ik)
    spath = spin_path(ik, ispin)
    bpath = os.path.join(spath, 'eigenvalues')
    band = fp[bpath].value
    bands[ik, :] = band
  return bands


# =======================================================================
