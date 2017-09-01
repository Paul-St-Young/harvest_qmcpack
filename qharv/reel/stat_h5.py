# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse hdf5 spectral and volumetric data output. Mostly built around h5py's API.

import os
import h5py

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

def mean_and_var(stat_fname,obs_path,nequil):
  """ calculate mean and variance of an observable from QMCPACK stat.h5 file

  assume autocorrelation = 1

  Args:
    stat_fname (str): .stat.h5 filename
    obs_path (str): path to observable, e.g. 'gofr_e_1_1'
    nequil (int): number of equilibration blocks to throw out
  Returns:
    (np.array,np.array): (val,var), the mean and variance of the observable
  """
  fp = h5py.File(stat_fname)
  if not obs_path in fp:
    raise RuntimeError('group %s not found' % obs_path)
  # end if

  val_path = os.path.join(obs_path,'value')
  valsq_path = os.path.join(obs_path,'value_squared')
  if not ((val_path in fp) and (valsq_path in fp)):
    raise RuntimeError('group %s must have "value" and "value_squared" to calculate variance' % obs_path)
  # end if

  val_data = fp[val_path].value
  nblock   = len(val_data)
  if (nequil>=nblock):
    raise RuntimeError('cannot throw out %d blocks from %d blocks'%(nequil,nblock))
  # end if
  val_mean = val_data[nequil:].mean(axis=0)

  valsq_data = fp[valsq_path].value
  var_mean = (valsq_data-val_data**2.)[nequil:].mean(axis=0)
  return val_mean,var_mean
# end def mean_and_var
