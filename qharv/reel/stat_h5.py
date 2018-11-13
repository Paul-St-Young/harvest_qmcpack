# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 spectral and volumetric data output. Mostly built around h5py's API.
import os
import h5py
import numpy as np

from qharv.seed.wf_h5 import read, ls
def path_loc(handle,path):
  return handle[path].value

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
  if kappa is None:
    try:
      from qharv.reel.forlib.stats import corr
    except:
      raise RuntimeError('please compile qharv.reel.forlib')
      #from qharv.reel.scalar_dat import corr  # slow Python implementation
    kappa = np.apply_along_axis(corr, axis, edata)
  neffective = len(edata)/kappa
  # calculate mean and error
  val_mean = edata.mean(axis=axis)
  val_std  = edata.std(ddof=1, axis=axis)
  val_err  = val_std/np.sqrt(neffective)
  return val_mean, val_err

def mean_and_err(handle,obs_path,nequil,kappa=None):
  """ calculate mean and variance of an observable from QMCPACK stat.h5 file

  assume autocorrelation = 1 by default

  Args:
    handle (h5py.Group): or h5py.File or h5py.Dataset
    obs_path (str): path to observable, e.g. 'gofr_e_1_1'
    nequil (int): number of equilibration blocks to throw out
    kappa (float,optional): auto-correlation of the data, default=1.0 i.e. no
     auto-correlation
  Returns:
    (np.array,np.array): (val_mean,val_err), the mean and error of the observable, assuming no autocorrelation. For correlated data, error is underestimated by a factor of sqrt(autocorrelation).
  """
  # look for hdf5 group corresponding to the requested observable
  if not obs_path in handle:
    raise RuntimeError('group %s not found' % obs_path)
  val_path = os.path.join(obs_path,'value')
  if not (val_path in handle):
    val_path = obs_path # !!!! assuming obs_path includes value already
    # `handle[val_path]` will fail if this assumption is not correct

  # get equilibrated data
  val_data = handle[val_path].value
  nblock   = len(val_data)
  if (nequil>=nblock):
    raise RuntimeError('cannot throw out %d blocks from %d blocks'%(nequil,nblock))
  edata      = val_data[nequil:] # equilibrated data

  # get statistics
  val_mean, val_err = me2d(edata)
  return val_mean,val_err

def absolute_magnetization(handle,nequil,obs_name='SpinDensity',up_name='u',dn_name='d'):
  """ calculate up-down spin density and the integral of its absolute value; first check that /SpinDensity/u and /SpinDensity/d both exist in the .stat.h5 file, then extract both densities, subtract and integrate

  Args:
    handle (h5py.Group): or h5py.File or h5py.Dataset
    nequil (int): number of equilibration blocks to throw out
    obs_name (str): name of spindensity observable, default=SpinDensity
    up_name (str): default=u
    dn_name (str): default=d
  Returns:
    (np.array,np.array,float,float): (rho,rhoe,mabs_mean,mabs_error), rho and rhoe have shape (ngrid,), where ngrid is the total number of real-space grid points i.e. for grid=(nx,ny,nz), ngrid=nx*ny*nz. mabs = \int |rho_up-rho_dn|
  """

  #print( ls(handle,r=True) ) # see what's in the file

  # make sure observable is in file
  if not obs_name in handle:
    raise RuntimeError('observable %s not found in %s' % (obs_name,stat_fname))
  # end if

  # make sure up and down components are in spin density
  up_loc = '%s/%s'%(obs_name,up_name)
  dn_loc = '%s/%s'%(obs_name,dn_name)
  if not ((up_loc in handle) and (dn_loc in handle)):
    raise RuntimeError('%s must have up and down components (%s,%s)' % (obs_name,up_name,dn_name))
  # end if

  # get up density
  val,err = mean_and_err(stat_fname,up_loc,nequil)
  rho_up  = val
  rho_up_err = err

  # get down density
  val,err = mean_and_err(stat_fname,dn_loc,nequil)
  rho_dn  = val
  rho_dn_err = err

  # get absolute difference
  rho  = rho_up-rho_dn
  rhoe = np.sqrt( rho_dn_err**2.+rho_up_err**2. )
  mabs_mean = abs(rho).sum()
  mabs_error= np.sqrt( (rhoe**2.).sum() )
  return rho,rhoe,mabs_mean,mabs_error
# end def absolute_magnetization

def dsk_from_csk(fp,csk_name,nequil,kappa,nsig=3):
  """ extract fluctuating structure factor dS(k) from charged structure factor S_c(k)
  Args:
    fp (h5py.File): stat.h5 handle
    csk_name (str): name the charged S(k) estimator, likely 'csk'
    nequil (int): equilibration length
    kappa (float): autocorrelation length
    nsig (int,optional): noise suppression level in units of standard error, default nsig=3, all entries below nsig are considered zero.
  Returns:
  """
  
  # extract raw S(k) data
  kvecs = path_loc(fp,'%s/kpoints/value'%csk_name)
  cskm0,cske0   = mean_and_err(fp,'%s/csk'%csk_name,nequil,kappa)
  crhom0,crhoe0 = mean_and_err(fp,'%s/crhok'%csk_name,nequil,kappa)

  # get <rho(k)>^2 and set noise to zero
  sel = abs(crhom0) < nsig*abs(crhoe0)
  crhom0[sel] = 0; crhoe0[sel] = 0
  crhom = np.sum(crhom0**2.,axis=0)

  # get <S(k)>
  cskm = cskm0.sum(axis=0)
  cske = np.sqrt( (cske0**2.).sum(axis=0) )

  # return <dS(k)>
  dskm = cskm - crhom
  dske = cske # !!!! assume errors in csk & crho are perfectly correlated
  return kvecs,dskm,dske
# end def dsk_from_csk

def gofr(fp,obs_name,nequil,kappa,force=False):
  """ extract pair correlation function g(r) from stat.h5 file
  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, should start with 'gofr', e.g. gofr_e_0_1
    nequil (int): number of equilibration blocks to remove
    kappa (float): correlation length
    force (bool,optional): force execution, i.e. skip all checks
  Returns:
    tuple: (myr,grm,gre): bin locations, g(r) mean, g(r) error
  """
  if (not obs_name.startswith('gofr')) and (not force):
    raise RuntimeError('%s does not start with "gofr"; set force=True to bypass'%obs_name)
  # end if

  grm,gre = mean_and_err(fp,'%s/value'%obs_name,nequil,kappa)
  rmax = path_loc(fp,'%s/cutoff'%obs_name)[0]
  dr   = path_loc(fp,'%s/delta'%obs_name)[0]

  # guess bin locations
  myr  = np.arange(0,rmax,dr)
  if (len(myr)!=len(grm)) and (not force):
    raise RuntimeError('num_bin mismatch; try read from input?')
  # end if
  return myr,grm,gre
# end def gofr

def nofk(fp,obs_name,nequil,kappa):
  """ extract momentum estimator output n(k) from stat.h5 file
  Args:
    fp (h5py.File): h5py handle of stat.h5 file
    obs_name (str): observable name, probably 'nofk'
    nequil (int): number of equilibration blocks to remove
    kappa (float): correlation length
    force (bool,optional): force execution, i.e. skip all checks
  Returns:
    tuple: (kvecs,nkm,nke): k-vectors, n(k) mean, n(k) error
  """
  kvecs = fp[obs_name]['kpoints'].value
  nofkm,nofke = mean_and_err(fp,'%s/value'%obs_name,nequil,kappa)
  return kvecs,nofkm,nofke
# end def nofk
