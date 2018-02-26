# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to collect QMCPACK data
#  building around qharv.reel, qharv.sieve
import os
import numpy as np
import pandas as pd

from qharv.reel  import mole,scalar_dat
from qharv.sieve import scalar_df

def collect_flist(flist,nequil,kappa,tmp_dat):
  """ collect all scalar.dat files in flist
  the search for scalar.dat is performed recursively
  temporary data are stored in ASCII format in tmp_dat
  there is one entry for each scalar.dat file

  * this routine should eventually superseded collect_mean_df

  Args:
    flist (list): a list of scalar.dat files to collect
    nequil (int): equilibration time
    kappa (float): autocorrelation time. If None is given, then kappa is calculated on-the-fly. The calculation of kappa is slow for a long trace.
  Returns:
    pd.DataFrame: one entry for each scalar.dat, containing mean and error of all columns. Two metadata columns ['path','fdat'] contain the path and filename of the scalar.dat, respectively.
  """

  if os.path.isfile(tmp_dat):
    raise RuntimeError('%s exists; delete to reanalyze'%tmp_dat)
  # end if

  if len(flist) == 0:
    raise RuntimeError('no scalar.dat given')
  # end if

  dat_path = os.path.dirname(tmp_dat)
  if not os.path.isdir(dat_path):
    sp.check_call(['mkdir',dat_path])
  # end if

  fp = open(tmp_dat,'w')
  data  = []
  ifloc = 0
  for floc in flist:
    # level 1: raw time series -> return a scalar_df
    mydf = scalar_dat.parse(floc)
    # level 2: average over time series -> return a mean_df
    mdf  = scalar_df.mean_error_scalar_df(mydf,nequil,kappa=kappa)
    assert len(mdf) == 1
    mdf['path'] = os.path.dirname(floc)
    mdf['fdat'] = os.path.basename(floc)

    # save processed data to file
    if ifloc == 0: 
      fp.write('# ' + '  '.join(mdf.columns) + '\n')
    # end if
    fp.write('  '.join(mdf.iloc[0].values.astype(str)) + '\n')

    # save processed data to memory
    data.append(mdf)
    ifloc += 1
  # end for
  fp.close()

  df = pd.concat(data).reset_index(drop=True)
  return df
# end def collect_flist
