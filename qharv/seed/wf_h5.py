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

def get(fp, name):  # see more advanced get at level 3
  """ retrieve data from a known location in pwscf.h5

  Args:
    fp (h5py.File): hdf5 file object
    name (str): a known name in locations
  Return:
    array_like: whatever fp[loc][()] returns
  """
  if name not in locations.keys():
    msg = 'unknown attribute requested: "%s"' % name
    msg += '\n known attributes:\n  ' + '\n  '.join(locations.keys())
    raise RuntimeError(msg)
  loc = locations[name]
  return fp[loc][()]

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
  elem_id  = fp['atoms/species_ids'][()]
  elem_map = {}
  nelem = fp['atoms/number_of_species'][()]
  for ielem in range(nelem):
    elem_name = fp['atoms/species_%d/name' % ielem][()][0]
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
  psig_arr = fp[orb_path][()]  # stored in real view
  # psig = psig_arr[:,0]+1j*psig_arr[:,1]  # convert to complex view
  psig = psig_arr.flatten().view(complex)  # more elegant conversion
  return psig

# ====== level 3: single particle orbitals ======

def get_cmat(fp, ikpt, ispin, norb=None, npw=None):
  """ get Kohn-Sham orbital coefficients on a list of plane waves (PWs)

  Args:
    fp (h5py.File): wf h5 file
    ikpt (int): kpoint index
    ispin (int): spin index
    norb (int, optional): number of orbitals at this kpoint
    npw (int, optional): number of PW for each orbital
  Return:
    np.array: cmat orbital coefficient matrix
  """
  # decide how many orbitals to extract (norb)
  if norb is None:
    nelecs = get(fp, 'nelecs')  # all spins
    norb = nelecs[ispin]  # get all occupied orbitals
  if npw is None:  # count the number of PWs (npw)
    gvecs = get(fp, 'gvectors')  # integer vectors
    npw = len(gvecs)
  # construct coefficient matrix
  cmat = np.zeros([norb, npw], dtype=complex)
  for iorb in range(norb):
    ci = get_orb_in_pw(fp, ikpt, ispin, iorb)
    cmat[iorb, :] = ci
  return cmat

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
    ukvec = fp[os.path.join(kpath, 'reduced_k')][()]
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
    band = fp[bpath][()]
    bands[ik, :] = band
  return bands

def get_orbs(fp, orbs, truncate=False, tol=1e-8):
  """ return the list of requested Kohn-Sham orbitals

  Args:
    fp (h5py.File): wf h5 file
    orbs (list): a list of 3-tuples, each tuple species the KS state
     by (kpoint/twist, spin, band) i.e. (ik, ispin, ib)
    truncate (bool, optional): remove PWs with ``small'' coefficient
    tol (float, optional): define ``small'' as |ck|^2 < tol
  """
  from qharv.inspect import axes_pos
  gvecs = get(fp, 'gvectors')
  qvecs = get_twists(fp)
  axes = get(fp, 'axes')
  raxes = axes_pos.raxes(axes)

  kvecsl = []
  psigl = []
  for orb in orbs:
    ik, ispin, ib = orb
    # PW basis
    kvecs = np.dot(gvecs+qvecs[ik], raxes)
    npw = len(kvecs)
    # PW coefficients
    psig = get_orb_in_pw(fp, ik, ispin, ib)
    sel = np.ones(npw, dtype=bool)
    if truncate:  # cut down on the # of PWs
      pg2 = (psig.conj()*psig).real
      sel = pg2 > tol
    kvecsl.append(kvecs[sel])
    psigl.append(psig[sel])
  return kvecsl, psigl

# ====== level 4: write wf h5 file from scratch ======

def write_kpoint(kgrp, ikpt, utvec, evals, cmats):
  """ fill the electrons/kpoint_$ikpt group in wf h5 file

  Args:
    kgrp (h5py.Group): kpoint group
    ikpt (int): twist index
    utvec (np.array): twist vector in reduced units
    evals (list): list of Kohn-Sham eigenvalues to sort orbitals;
      one real np.array of shape (norb) for each spin
    cmats (list): list of Kohn-Sham orbitals in PW basis;
      one complex np.array of shape (norb, npw) for each spin

  Example:
    >>> fp = h5py.File('pwscf.pwscf.h5', 'w')
    >>> kgrp = fp.create_group('/electrons/kpoint_0')
    >>> evals = [ np.array([0]) ]  # 1 spin, 1 state
    >>> cmats = [ np.array([[0]], dtype=complex) ]
    >>> write_kpoint(kgrp, 0, [0, 0, 0], evals, cmats)
    >>> fp.close()
  """
  # write twist
  kgrp.create_dataset('reduced_k', data=utvec)
  # write Kohn-Sham system
  nspin = len(cmats)
  if len(evals) != nspin:
    raise RuntimeError('%d evals %d cmats' % (len(evals), nspin))
  for ispin, (evs, cmat) in enumerate(zip(evals, cmats)):
    spath = 'spin_%d' % ispin
    sgrp = kgrp.create_group(spath)
    # write eigenvalues
    nstate = len(cmat)
    if len(evs) != nstate:
      raise RuntimeError('%d evs for %d states' % (len(evs), nstate))
    sgrp.create_dataset('number_of_states', data=[nstate])
    sgrp.create_dataset('eigenvalues', data=evs)
    # write eigenvectors
    for istate, evec in enumerate(cmat):
      psi_g_path = '%s/state_%d/psi_g' % (spath, istate)
      real_emat = np.zeros([len(evec), 2])
      real_emat[:, 0] = evec.real
      real_emat[:, 1] = evec.imag
      pgrp = kgrp.create_dataset(psi_g_path, data=real_emat)
  # no symmetry infomation
  kgrp.create_dataset('numsym', data=[1])
  kgrp.create_dataset('symgroup', data=[1])
  kgrp.create_dataset('weight', data=[1])

def write_wf(egrp, utvecs, gvecs, evalsl, cmatsl):
  """ fill the wf portion of the electrons group in wf h5 file
  !!!! WARNING: this function may require too much memory;
  if so, use write_kpoint directly

  Args:
    egrp (h5py.Group): electrons group
    utvecs (np.array): all twist vectors in reduced units
    evalsl (list): list of Kohn-Sham eigenvalues to sort orbitals;
      one real np.array of shape (norb) for each spin and twist
    cmatsl (list): list of Kohn-Sham orbitals in PW basis;
      one complex np.array of shape (norb, npw) for each spin and twist
  """
  npw0 = len(gvecs)  # number of PWs
  # create kpoints
  kp_fmt = 'kpoint_%d'
  for ik, (utvec, evals, cmats) in enumerate(
      zip(utvecs, evalsl, cmatsl)
  ):
    # check PW count
    for ispin in range(len(cmats)):
      npw = cmats[ispin].shape[1]
      if npw != npw0:
        raise RuntimeError('k%d has %d PW, not %d gvecs' % (ik, npw, npw0))
    # create and fill kpoint group
    kpath = kp_fmt % ik
    kgrp = egrp.create_group(kpath)
    write_kpoint(kgrp, ik, utvec, evals, cmats)
  # add gvectors to kpoint0
  kpath0 = kp_fmt % 0
  kgrp0 = egrp[kpath0]
  kgrp0.create_dataset('gvectors', data=gvecs)
  kgrp0.create_dataset('number_of_gvectors', data=[len(gvecs)])

# =======================================================================
