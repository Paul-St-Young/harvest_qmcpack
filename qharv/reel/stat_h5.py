# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 spectral and volumetric data output. Mostly built around h5py's API.
import os
import h5py
import numpy as np

def read(stat_fname):
  return h5py.File(stat_fname)
def path_loc(path,handle):
  return handle[path].value

def ls(handle,r=False,level=0,indent="  "):
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
  mystr=''
  if isinstance(handle,h5py.File) or isinstance(handle,h5py.Group):
    for key,val in handle.items():
      mystr += indent*level+'/'+key + "\n"
      if r:
        mystr += ls(val,r=r,level=level+1,indent=indent)
    # end for
  elif isinstance(handle,h5py.Dataset):
    return ''
  else:
    raise RuntimeError('cannot handle type=%s'%type(handle))
  # end if
  return mystr
# end def ls

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
  if not obs_path in handle:
    raise RuntimeError('group %s not found' % obs_path)
  # end if
  if kappa is None:
    raise NotImplementedError('need an automatic way to calculate auto-correlation')
  # end if

  val_path = os.path.join(obs_path,'value')
  if not (val_path in handle):
    val_path = obs_path # !!!! assuming obs_path includes value already
    # `handle[val_path]` will fail if this assumption is not correct
  # end if

  val_data = handle[val_path].value
  nblock   = len(val_data)
  if (nequil>=nblock):
    raise RuntimeError('cannot throw out %d blocks from %d blocks'%(nequil,nblock))
  # end if
  edata      = val_data[nequil:] # equilibrated data
  neffective = (nblock-nequil)/kappa

  # calculate mean and error
  val_mean = edata.mean(axis=0)
  val_std  = edata.std(ddof=1,axis=0)
  val_err  = val_std/np.sqrt(neffective)
  
  return val_mean,val_err
# end def mean_and_err

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
