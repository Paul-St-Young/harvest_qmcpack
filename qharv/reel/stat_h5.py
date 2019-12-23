# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 spectral and volumetric data output.
#  Mostly built around h5py's API.
import os
import h5py
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
    try:  # fortran implementation is faster for len(trace)<1000
      from qharv.reel.forlib.stats import corr
    except ImportError:  # numpy FFT scales better to long traces
      if ntrace < 1024:
        msg = 'using slow Python implementation'
        msg += ' please compile qharv.reel.forlib'
        print(msg)
      from qharv.reel.scalar_dat import corr
    kappa = np.apply_along_axis(corr, axis, edata)
  neffective = ntrace/kappa
  # calculate mean and error
  val_mean = edata.mean(axis=axis)
  val_std  = edata.std(ddof=1, axis=axis)
  val_err  = val_std/np.sqrt(neffective)
  return val_mean, val_err

def mean_and_err(handle, obs_path, nequil, kappa=None):
  """ calculate mean and variance of an observable from QMCPACK stat.h5 file

  assume autocorrelation = 1 by default

  Args:
    handle (h5py.Group): or h5py.File or h5py.Dataset
    obs_path (str): path to observable, e.g. 'gofr_e_1_1'
    nequil (int): number of equilibration blocks to throw out
    kappa (float,optional): auto-correlation of the data, default=1.0 i.e. no
     auto-correlation
  Returns:
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
    kappa (float, optional): autocorrelation length, default is to calcaulte
     on-the-fly
  Return:
    (np.array, np.array, np.array): (kvecs, dskm, dske), kvectors and S(k)
    mean and error
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
    kappa (float, optional): autocorrelation, default is to calculate
     on-the-fly
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
  return kvecs, rhokm.view(float), rhoke.view(float)

def gofr(fp, obs_name, nequil, kappa=None, force=False):
  """ extract pair correlation function g(r) from stat.h5 file
  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, should start with 'gofr', e.g. gofr_e_0_1
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): autocorrelation length, default is to calcaulte
     on-the-fly
    force (bool,optional): force execution, i.e. skip all checks
  Returns:
    tuple: (myr,grm,gre): bin locations, g(r) mean, g(r) error
  """
  if (not obs_name.startswith('gofr')) and (not force):
    msg = '%s does not start with "gofr"; set force=True to bypass' % obs_name
    raise RuntimeError(msg)

  grm, gre = mean_and_err(fp, '%s/value' % obs_name, nequil, kappa)
  rmax = path_loc(fp, '%s/cutoff' % obs_name)[0]
  dr   = path_loc(fp, '%s/delta' % obs_name)[0]

  # guess bin locations
  myr  = np.arange(0, rmax, dr)
  if (len(myr) != len(grm)) and (not force):
    raise RuntimeError('num_bin mismatch; try read from input?')
  return myr, grm, gre

def nofk(fp, obs_name, nequil, kappa=None):
  """ extract momentum estimator output n(k) from stat.h5 file

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably 'nofk'
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): autocorrelation, default is to calculate
     on-the-fly
  Return:
    (np.array, np.array, np.array): (kvecs,nkm,nke) k-vectors, n(k) mean and
     error
  """
  kvecs = fp[obs_name]['kpoints'][()]
  nkm, nke = mean_and_err(fp, '%s/value' % obs_name, nequil, kappa)
  return kvecs, nkm, nke

# ---------------- begin charged structure factor ----------------
# note: charged S(k), rho(k) were added for multi-component simulation,
#  e.g. electrons & protons. Nobody does mc sim. so deprecate?
def dsk_from_csk(fp, csk_name, nequil, kappa=None):
  """ extract fluctuating structure factor dS(k) from charged structure factor

  Args:
    fp (h5py.File): stat.h5 handle
    csk_name (str): name the charged S(k) estimator, likely 'csk'
    nequil (int): equilibration length
    kappa (float, optional): autocorrelation length, default is to calcaulte
     on-the-fly
  Return:
    (np.array, np.array, np.array): (kvecs, dskm, dske), kvectors and S(k)
    mean and error
  """
  # get data
  kpt_path = '%s/kpoints/value' % csk_name
  csk_path = '%s/csk/value' % csk_name
  crho_path = '%s/crhok/value' % csk_name
  kvecs = fp[kpt_path][()]
  cska = fp[csk_path][()]
  crhoa = fp[crho_path][()]
  nblock, nspin, nk = cska.shape

  # get dsk using equilibrated data
  dska = cska[nequil:] - crhoa[nequil:]**2
  dskm, dske = me2d(dska)
  return kvecs, dskm, dske

def rhok(fp, obs_name, nequil, kappa=None):
  """ extract electronic density rho(k) from stat.h5 file

  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably 'csk'
    nequil (int): number of equilibration blocks to remove
    kappa (float, optional): autocorrelation, default is to calculate
     on-the-fly
  Return:
    (np.array, np.array, np.array): (kvecs, rhokm, rhoke)
      k-vectors, rho(k) mean and error
      notice rhokm has two rows for real and imag components (2, nk)
  """
  # get data
  kpt_path = '%s/kpoints/value' % obs_name
  crho_path = '%s/crhok/value' % obs_name
  kvecs = fp[kpt_path][()]
  rhokm, rhoke = mean_and_err(fp, crho_path, nequil, kappa)
  return kvecs, rhokm, rhoke
# ---------------- charged structure factor end ----------------
