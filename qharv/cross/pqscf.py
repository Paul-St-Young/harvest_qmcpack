# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate pyscf results for use in QMCPACK
import numpy as np

# ======================= level 0: basic pyscf ======================

def mf_from_chkfile(chkfile, scf_class=None, pbc=True):
  if pbc:
    from pyscf.pbc import scf
  else:
    from pyscf import scf
  if scf_class is None:
    scf_class = scf.RHF
  cell, scf_rec = scf.chkfile.load_scf(chkfile)
  mf = scf_class(cell)
  mf.__dict__.update(scf_rec)
  return mf

def is_uhf(mo_coeff):
  return not np.isscalar(mo_coeff[0][0])

def reorder(mf, order, ispin=None):
  if is_uhf(mf.mo_coeff):  # UHF
    if ispin is None:
      raise RuntimeError('must provide ispin for UHF')
    coeff = mf.mo_coeff[ispin]
    evals = mf.mo_energy[ispin]
  else:  # RHF or ROHF
    coeff = mf.mo_coeff
    evals = mf.mo_energy
  old_coeff = coeff.copy()
  old_energy = evals.copy()
  for i, j in enumerate(order):
    coeff[:, i] = old_coeff[:, j]
    evals[i] = old_energy[j]

def get_ao_symm(mol):
  from pyscf import symm
  names = mol.irrep_name
  orbs = mol.symm_orb
  nao = mol.nao
  return symm.label_orb_symm(mol, names, orbs, np.eye(nao))

def show_orbital_occupations(mol, mf, nshow):
  from pyscf import symm
  # Q/ how many s basis?
  names = mol.irrep_name
  orbs = mol.symm_orb
  nao = mol.nao
  aosym = get_ao_symm(mol)
  ns = 0
  for sym in aosym:
    if sym != 'Ag':
      break
    ns += 1
  # A/ ns
  print()
  print('# MO schar occ  eigenvalue')
  sfmt = '%4s %5.2f %3.1f %10.6f'
  if is_uhf(mf.mo_coeff):  # UHF
    ev1 = mf.mo_energy[0]
    co1 = mf.mo_coeff[0]
    oc1 = mf.mo_occ[0]
    ev2 = mf.mo_energy[1]
    co2 = mf.mo_coeff[1]
    oc2 = mf.mo_occ[1]
    os1 = symm.label_orb_symm(mol, names, orbs, co1)
    os2 = symm.label_orb_symm(mol, names, orbs, co2)
    if mf.mo_energy == 'None':  # CASSCF
      ev1 = ev2 = -np.ones(nshow+1)
    i = 0
    for e1, o1, s1, c1, e2, o2, s2, c2 in zip(
      ev1[:nshow+1], oc1, os1, co1.T, ev2, oc2, os2, co2.T
    ):
      schar1 = np.dot(c1[:ns], c1[:ns].conj()).real
      schar2 = np.dot(c2[:ns], c2[:ns].conj()).real
      sup = sfmt % (s1, schar1, o1, e1)
      sdn = sfmt % (s2, schar2, o2, e2)
      print('%s %s %d' % (sup, sdn, i))
      i += 1
  else:  # RHF or ROHF
    evals = mf.mo_energy
    coeff = mf.mo_coeff
    occ = mf.mo_occ
    orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, coeff)
    i = 0
    for e, o, s, c in zip(evals[:nshow+1], occ, orbsym, coeff.T):
      sup = sfmt % (s, sum(c[:ns]), o, e)
      print('%s %d' % (sup, i))
      i += 1

def atom_text(elem, pos):
  """convert elem,pos to text representation

  for example, elem = ['C','C'], pos = [[0,0,0],[0.5,0.5,0.5]] will be
  converted to 'C 0 0 0;C 0.5 0.5 0.5'

  Args:
   elem (list): a list of atomic symbols such as 'H','C','O'
   pos  (list): a list of atomic positions, assume in 3D
  Returns:
   str: atomic string accepted by pyscf"""
  assert len(elem) == len(pos)
  lines = []
  for iatom in range(len(elem)):
      mypos = pos[iatom]
      assert len(mypos) == 3
      line = '%5s  %10.6f  %10.6f  %10.6f' % (
        elem[iatom], mypos[0], mypos[1], mypos[2])
      lines.append(line)
  atext = ';\n'.join(lines)
  return atext

# ======================= level 1: model hamil ======================

def read_model(fh5):
  import h5py
  fp = h5py.File(fh5, 'r')
  h0 = fp['h0'][()][0]
  h1 = fp['h1'][()]
  h2 = fp['h2'][()]
  order = fp['h2_index_order'][()][0].decode()
  fp.close()
  if order == "physicist":
    eri = h2.transpose(0, 2, 1, 3).copy()
  elif order == "chemist":
    eri = h2
  else:
    msg = 'unknown ERI index order "%s"' % order
    raise RuntimeError(msg)
  return h0, h1, eri

def define_model(nelec, h1, eri, h0=0, ovlp=None, restore=False, scf_class=None):
  from scipy.linalg import block_diag
  from pyscf import gto, scf, ao2mo
  nmo = len(h1)
  if ovlp is None:
    ovlp = np.eye(nmo)
  if scf_class is None:
    scf_class = scf.UHF
  # define nelec
  mol = gto.M()
  mol.incore_anyway = True
  mol.nelec = nelec  # (nup, ndn)
  mol.tot_electrons = lambda *args: sum(nelec)
  mol.energy_nuc = lambda *args: float(h0)
  # define model
  mf = scf_class(mol)
  mf._eri = ao2mo.restore(1, eri, nmo) if restore else eri
  if isinstance(mf, scf.ghf.GHF):
    mf.get_hcore = lambda *args: block_diag(h1, h1)
    mf.get_ovlp = lambda *args: block_diag(ovlp, ovlp)
  else:
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: ovlp
  return mf

def smear_to_converge(mf, dm0, smears=None, method="fermi"):
  from pyscf.pbc.scf.addons import smearing_
  if smears is None:
    smears = [1e-1, 1e-3, 1e-5]
  for smear in smears:
    mf1 = smearing_(mf, sigma=smear, method=method)
    mf1.run(dm0)
    dm1 = mf1.make_rdm1()
    dm0 = dm1
  return mf1

# ======================== level 1: structure =======================

def ase_tile(cell, tmat):
  """Create supercell from primitive cell and tiling matrix

  Args:
    cell (pyscf.Cell): cell object
    tmat (np.array): 3x3 tiling matrix e.g. 2*np.eye(3)
  Return:
    pyscf.Cell: supercell
  """
  try:
    from qharv.inspect.axes_elem_pos import ase_tile as atile
  except ImportError:
    msg = 'tiling with non-diagonal matrix require the "ase" package'
    raise RuntimeError(msg)
  # get crystal from cell object
  axes = cell.lattice_vectors()
  elem = [atom[0] for atom in cell._atom]
  pos = cell.atom_coords()
  axes1, elem1, pos1 = atile(axes, elem, pos, tmat)
  # re-make cell object
  cell1 = cell.copy()
  cell1.atom = list(zip(elem1, pos1))
  cell1.a = axes1
  # !!!! how to change mesh ????
  ncopy = np.diag(tmat)
  cell1.mesh = np.array([ncopy[0]*cell.mesh[0],
                         ncopy[1]*cell.mesh[1],
                         ncopy[2]*cell.mesh[2]])
  cell1.build(False, False, verbose=0)
  cell1.verbose = cell.verbose
  return cell1

# ======================== level 2: orbital =========================

def check_grid_shape(grid_shape, gvecs):
  # Nyquist-Shannon sampling grid
  ns_shape = 2*(gvecs.max(axis=0)-gvecs.min(axis=0))+1
  if grid_shape is None:
    # deduce minimum real-space basis to retain all information
    grid_shape = ns_shape
  else:  # make sure no information is lost from pw representation
    if not (grid_shape >= ns_shape).all():
      msg = 'grid shape %s is too small to preserve PW rep.' % str(grid_shape)
      msg += 'Please increase to at least %s' % str(ns_shape)
      raise RuntimeError(msg)
  return grid_shape

def pw_to_r(gvecs, psig, grid_shape=None):
  """ convert a function from plane-wave to real-space basis

  plane wave basis is assumed to be in reciprocal-lattice units
  real-space basis will be in grid units axes/grid_shape

  Args:
    gvecs (np.array):  dtype=int & shape = (npw,ndim), npw is the number of plane waves, and ndim is the number of spatial dimensions. ndim is expected to be 3. Each entry in gvecs should be a 3D vector of integers.  Each gvec specify a plane wave basis element exp(i*gvec*rvec).
    psig (np.array): dtype=complex & shape = (npw,). Expansion coefficients in PW basis.
    grid_shape (np.array): dtype=int & shape = (ndim,). Shape of real-space grid.
  Returns:
    (np.array,np.array): (grid_shape,moR), grid_shape is input if given. Otherwise constructed in function to hold all information from the plane-wave representation. moR has dtype=complex & shape = grid_shape. moR is the 3D function in real-space basis. Typically a molecular orbital.
  """
  # verify user input
  gs = check_grid_shape(grid_shape, gvecs)
  # perform Fourier transform
  npw, ndim = gvecs.shape
  fftbox = np.zeros(gs, dtype=complex)
  for ig in range(npw):
    fftbox[tuple(gvecs[ig])] = psig[ig]
  rgrid = np.fft.ifftn(fftbox)
  return gs, rgrid

def r_to_pw(moR0, grid_shape, gvecs=None):
  """ convert a 3D function from real-space to plane-wave basis

  This function is essentially the inverse of pw_to_r, but assumes that the real space grid is built around (0,0,0)

  Args:
    moR0 (np.array): dtype=complex & shape = (ngrid,), ngrid is the number of grid points. 3D function in real-space grid basis.
    grid_shape (np.array): dtype=int & shape = (ndim,). Shape of real-space grid.
  Returns:
    (np.array,np.array): (gvecs,psig), gvecs is input if given. Otherwise internally constructed. psig has dtype=complex & shape = (npw,). psig is the 3D function in plane-wave basis. Typically a molecular orbital.
  """
  assert np.prod(grid_shape) == len(moR0)
  cell_gs = (grid_shape-1)/2

  if gvecs is None:
    # deduce minimum plane-wave basis to retain all information
    if not (2*cell_gs+1 == grid_shape).all():
      msg = 'Please provide grid_shape. I cannot deduce minimum PW basis'
      msg += ' for even grid_shape %s' % str(grid_shape)
      raise RuntimeError(msg)

    nx, ny, nz = cell_gs
    from itertools import product
    gvecs = np.array([gvec for gvec in product(
      range(-nx, nx+1), range(-ny, ny+1), range(-nz, nz+1))], dtype=int)
  else:  # make sure information is not missing from real-space representation
    assert np.issubdtype(gvecs.dtype, np.integer)
    valid = (gvecs.max(axis=0) <= cell_gs).all() and \
            (-gvecs.min(axis=0) <= cell_gs).all()
    if not valid:
      msg = 'Please remove gvectors outside of cell_gs: %s' % str(cell_gs)
      raise RuntimeError(msg)

  npw, ndim = gvecs.shape
  assert ndim == 3

  orb = moR0.reshape(grid_shape)
  moG = np.fft.fftn(orb)/np.prod(grid_shape)

  psig = np.zeros(npw, dtype=complex)
  for ipw in range(npw):
    psig[ipw] = moG[tuple(gvecs[ipw])]
  return gvecs, psig

# ========================== level 2: spin ==========================

def calc_spin_sph(swf):
  snorm = np.dot(swf.conj(), swf)**0.5
  chi = swf/snorm
  rho1, rho2 = (chi.conj()*chi)**0.5
  theta = 2*np.arccos(rho1)
  assert np.isclose(rho1, np.cos(theta/2))
  assert np.isclose(rho2, np.sin(theta/2))
  phi1, phi2 = np.angle(chi)
  phi = phi2-phi1
  return snorm, theta, phi

def rotate_spin(theta, phi):
  # Shankar (14.3.47)
  #  rotate from \hat{z} = (0, 0) to \hat{n} = (\theta, \phi)
  rmat = np.array([
    [np.cos(theta/2), -np.sin(theta/2)*np.exp(-1j*phi)],
    [np.sin(theta/2)*np.exp(1j*phi), np.cos(theta/2)],
  ])
  return rmat

def calc_sz(mf):
  from pyscf import scf
  sigz = np.array([[1, 0], [0, -1]])/2
  if isinstance(mf, scf.hf.RHF):
    nup = sum(mf.mo_occ > 0)
    ndn = sum(mf.mo_occ > 1)
    sz = (nup-ndn)/2
  elif isinstance(mf, scf.uhf.UHF):
    mo_a, mo_b = [mf.mo_coeff[i][:, mf.mo_occ[i]>0] for i in range(2)]
    nup = mo_a.shape[1]
    ndn = mo_b.shape[1]
    sz = (nup-ndn)/2
  elif isinstance(mf, scf.ghf.GHF):
    nao = len(mf.mo_coeff)//2
    mo1 = mf.mo_coeff[:, mf.mo_occ>0]
    nmo = mo1.shape[1]
    sz = 0
    for imo in range(nmo):
      swf = np.array([mo1[:nao, imo].sum(), mo1[nao:, imo].sum()])
      snorm = np.dot(swf.conj(), swf)**0.5
      sz1 = swf.conj().T@sigz@swf
      sz += sz1
  else:
    msg = 'unknown type "%s"' % type(mf)
    raise RuntimeError(msg)
  return sz
