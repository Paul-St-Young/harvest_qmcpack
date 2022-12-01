# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 spectral and volumetric data output.
#  Mostly built around h5py's API.
import os
import numpy as np

from qharv.seed.wf_h5 import read, ls

def path_loc(handle, path):
  return handle[path][()]

def me2d(edata, kappa=None, axis=0):
  """ Calculate mean and error of a table of columns

  Args:
    edata (np.array): 2D array of equilibrated time series data
    kappa (float, optional): pre-calculate auto-correlation, default is to
     re-calculate on-the-fly
    axis (int, optional): axis to average over, default 0 i.e. columns

  Return:
    (np.array, np.array): (mean, error) of each column
  """
  # get autocorrelation
  ntrace = edata.shape[axis]
  if kappa is None:
    try:  # fortran implementation is faster than np FFT for len(trace)<1000
      from qharv.reel.forlib.stats import corr
    except ImportError as err:
      msg = str(err)
      msg += '\n  Please compile qharv.reel.forlib.stats using f2py.'
      raise ImportError(msg)
    kappa = np.apply_along_axis(corr, axis, edata.real)
  neffective = ntrace/kappa
  # calculate mean and error
  val_mean = edata.mean(axis=axis)
  val_std  = edata.std(ddof=1, axis=axis)
  val_err  = val_std/np.sqrt(neffective)
  return val_mean, val_err

def mean_and_err(handle, obs_path, nequil, kappa=None):
  """ calculate mean and error of an observable from QMCPACK stat.h5 file

  Args:
    handle (h5py.Group): or h5py.File or h5py.Dataset
    obs_path (str): path to observable, e.g. 'gofr_e_1_1'
    nequil (int): number of equilibration blocks to throw out
    kappa (float, optional): auto-correlation, default recalculate

  Return:
    (np.array, np.array): (mean, err), the mean and error of observable
  """
  # look for hdf5 group corresponding to the requested observable
  if obs_path not in handle:
    raise RuntimeError('group %s not found' % obs_path)
  val_path = os.path.join(obs_path, 'value')
  if not (val_path in handle):
    val_path = obs_path  # !!!! assuming obs_path includes value already
    # `handle[val_path]` will fail if this assumption is not correct

  # get equilibrated data
  val_data = handle[val_path][()]
  nblock   = len(val_data)
  if (nequil >= nblock):
    msg = 'cannot throw out %d blocks from %d blocks' % (nequil, nblock)
    raise RuntimeError(msg)
  edata = val_data[nequil:]

  # get statistics
  val_mean, val_err = me2d(edata)
  return val_mean, val_err

def dsk_from_skall(fp, obs_name, nequil, kappa=None):
  """ extract fluctuating structure factor dS(k) from skall

  Args:
    fp (h5py.File): stat.h5 handle
    obs_name (str, optional): name the "skall" estimator, likely "skall"
    nequil (int): equilibration length
    kappa (float, optional): auto-correlation, default recalculate

  Return:
    (np.array, np.array, np.array): (kvecs, dskm, dske),
     kvectors and S(k) mean and error
  """
  # get data
  kpt_path = '%s/kpoints/value' % obs_name
  sk_path = '%s/rhok_e_e/value' % obs_name
  rhoki_path = '%s/rhok_e_i/value' % obs_name
  rhokr_path = '%s/rhok_e_r/value' % obs_name
  kvecs = fp[kpt_path][()]
  ska = fp[sk_path][()]
  rkra = fp[rhokr_path][()]
  rkia = fp[rhoki_path][()]
  nblock, nk = ska.shape
  assert len(kvecs) == nk
  dska = ska[nequil:]-(rkra[nequil:]**2+rkia[nequil:]**2)
  dskm, dske = me2d(dska)
  return kvecs, dskm, dske

def rhok_from_skall(fp, obs_name, nequil, kappa=None):
  """ extract electronic density rho(k) from stat.h5 file

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str, optional): name the "skall" estimator, likely "skall"
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): auto-correlation, default recalculate

  Return:
    (np.array, np.array, np.array): (kvecs, rhokm, rhoke)
      k-vectors, rho(k) mean and error, shape (nk, ndim)
      notice rhok is the real-view of a complex vector, shape (2*nk,)
  """
  # get data
  kpt_path = '%s/kpoints/value' % obs_name
  rhokr_path = '%s/rhok_e_r/value' % obs_name
  rhoki_path = '%s/rhok_e_i/value' % obs_name
  kvecs = fp[kpt_path][()]
  rkrm, rkre = mean_and_err(fp, rhokr_path, nequil, kappa)
  rkim, rkie = mean_and_err(fp, rhoki_path, nequil, kappa)
  rhokm = rkrm + 1j*rkim
  rhoke = rkre + 1j*rkie
  return kvecs, rhokm, rhoke

def gofr(fp, obs_name, nequil, kappa=None, force=False):
  """ extract pair correlation function g(r) from stat.h5 file
  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, should start with 'gofr', e.g. gofr_e_0_1
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): auto-correlation, default recalculate
    force (bool, optional): force execution, i.e. skip all checks

  Returns:
    tuple: (myr, grm, gre): bin locations, g(r) mean, g(r) error
  """
  if (not obs_name.startswith('gofr')) and (not force):
    msg = '%s does not start with "gofr"; set force=True to bypass' % obs_name
    raise RuntimeError(msg)

  grm, gre = mean_and_err(fp, '%s/value' % obs_name, nequil, kappa)
  try:
    rmax = path_loc(fp, '%s/cutoff' % obs_name)[0]
  except IndexError:
    rmax = path_loc(fp, '%s/cutoff' % obs_name)
  try:
    dr   = path_loc(fp, '%s/delta' % obs_name)[0]
  except IndexError:
    dr   = path_loc(fp, '%s/delta' % obs_name)

  # guess bin locations
  myr  = np.arange(0, rmax-dr/10, dr)
  if (len(myr) != len(grm)) and (not force):
    msg = 'guess %d, but found %d bins\n' % (len(myr), len(grm))
    msg += ' need to fix guess.'
    raise RuntimeError(msg)
  return myr, grm, gre

def nofk(fp, obs_name, nequil, kappa=None):
  """ extract momentum estimator output n(k) from stat.h5 file

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably 'nofk'
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): auto-correlation, default recalculate

  Return:
    (np.array, np.array, np.array): (kvecs, nkm, nke),
     k-vectors, n(k) mean and error
  """
  kvecs = fp[obs_name]['kpoints'][()]
  nkm, nke = mean_and_err(fp, '%s/value' % obs_name, nequil, kappa)
  return kvecs, nkm, nke

def rdm1(fp, obs_name, nequil, kappa=None):
  """ extract 1RMD output from stat.h5 file

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably '1rdms'
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): auto-correlation, default recalculate

  Return:
    dict: a dictionary of 1RDMs, one for each species (eg. u, d)
  """
  matrix_path = os.path.join(obs_name, 'number_matrix')
  groups = fp[matrix_path].keys()
  rdms = {}
  for grp in groups:
    path = os.path.join(matrix_path, grp, 'value')
    ym, ye = mean_and_err(fp, path, nequil, kappa)
    rdms[grp] = (ym, ye)
  return rdms

def afobs(fp, obs_name, nequil, kappa=None, group='BackPropagated', numer='one_rdm', iav=None):
  """ extract 1RMD output from AFQMC stat.h5 file
   assume BackPropagated (BP) 'Observables/BackPropagated'

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably 'FullOneRDM'
    nequil (int): number of equilibration BP blocks to remove
    kappa (float, optional): auto-correlation, default recalculate
    numer (str, optional): numerator to extract, default 'one_rdm'
    iav (int, optional): BP level (Average_$iav), default is last level
  Return:
    tuple: (mean, error) arrays
  """
  # 1. gather meta data
  meta_paths = {
    'walker_type': 'Metadata/WalkerType',
    'nmo': 'Metadata/NMO',
    'dt': 'Metadata/Timestep',
    'free_projection': 'Metadata/FreeProjection',
  }
  meta = {}
  for key, path in meta_paths.items():
    meta[key] = fp[path][()]
  if meta['free_projection'] > 0:
    msg = 'need to consider denominator!'
    raise NotImplementedError(msg)
  nbas = int(meta['nmo'])
  itwalker = int(meta['walker_type'])
  if itwalker == 1:  # CLOSED
    rdm_shape = (1, nbas, nbas)
  elif itwalker == 2:  # COLLINEAR
    rdm_shape = (2, nbas, nbas)
  elif itwalker == 3:  # non-collinear
    rdm_shape = (1, 2*nbas, 2*nbas)
  else:
    msg = 'unknown walker type %d' % itwalker
    raise RuntimeError(msg)
  # 2. deal with back propagation (BP)
  avg_path = os.path.join('Observables', group, obs_name)
  if iav is None:  # use longest BP
    avgs = fp[avg_path].keys()
    iavgs = [int(a.replace('Average_', '')) for a in avgs]
    mav = max(iavgs)
  else:  # use user request
    mav = iav
  matrix_path = os.path.join(avg_path, 'Average_%d' % mav)
  # 3. get 1RDM at all equilibrated blocks
  blocks = fp[matrix_path].keys()
  rdm_blocks = [key for key in blocks if key.startswith(numer)]
  nblock = len(rdm_blocks)
  if nequil >= nblock:
    msg = 'cannot discard %d/%d blocks' % (nequil, nblock)
    raise RuntimeError(msg)
  data = []
  for block in rdm_blocks[nequil:]:
    path = os.path.join(matrix_path, block)
    rdm = fp[path][()].view(np.complex128)
    dpath = os.path.join(matrix_path, block.replace(numer, 'denominator'))
    deno = fp[dpath][()].view(np.complex128)
    data.append(rdm/deno)
  assert np.prod(rdm_shape) == np.prod(rdm.shape)
  # 4. get mean and standard error
  mat = np.array(data, dtype=np.complex128).reshape(
    -1, np.prod(rdm_shape))
  ym, ye = me2d(mat)
  dm = ym.reshape(rdm_shape)
  de = ye.reshape(rdm_shape)
  if itwalker == 3:  # non-collinear
    dm = np.array([
      dm[0, :nbas, :nbas], dm[0, nbas:, nbas:],
      dm[0, :nbas, nbas:], dm[0, nbas:, :nbas],
    ])
    de = np.array([
      de[0, :nbas, :nbas], de[0, nbas:, nbas:],
      de[0, :nbas, nbas:], de[0, nbas:, :nbas],
    ])
  return dm, de
