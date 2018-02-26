# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to roughly process scalar Dataframes. Mostly built around pandas's API.
#
#  note: A scalar dataframe (scalar_df) is expected to contain the raw data, i.e.
# block-resolved expectation values, of a SINGLE calculation. If multiple runs are
# collected in the same dataframe, label by ['path','fdat'] and use groupby before
# applying the functions in this script.
import numpy as np
import pandas as pd
from qharv.reel.scalar_dat import error

def mean_error_scalar_df(df,nequil,kappa=None):
  """ get mean and average from a dataframe of raw scalar data (per-block) 
   take dataframe having columns ['LocalEnergy','Variance',...] to a 
   dataframe having columns ['LocalEnergy_mean','LocalEnergy_error',...]

   Args:
    df (pd.DataFrame): raw scalar dataframe, presumable generated using
     qharv.scalar_dat.parse with extra labels columns added to identify
     the different runs.
    nequil (int): number of equilibration blocks to throw out for each run.
   Returns:
    pd.DataFrame: mean_error dataframe 
  """
  sel = df['index'] >= nequil # zero indexing

  # create pd.Series of mean
  msr = df.loc[sel].apply(np.mean).drop('index')

  # create pd.Series of error
  efunc = lambda x:error(x,kappa=kappa) # if kappa is not None, then 
  # auto-correlation is not re-calculated
  esr = df.loc[sel].apply( # error cannot be directly applied to matrix yet
    lambda x:float( np.apply_along_axis(efunc,0,x) )
  ).drop('index')

  df1 = msr.to_frame().T
  df2 = esr.to_frame().T
  jdf = df1.join(df2,lsuffix='_mean',rsuffix='_error')
  return jdf
# end def

def reblock(trace,block_size,min_nblock=4):
  """ block scalar trace to remove autocorrelation; see usage example in reblock_scalar_df
  Args:
    trace (np.array): a trace of scalars, may have multiple columns !!!! assuming leading dimension is the number of current blocks.
    block_size (int): size of block in units of current block.
    min_nblock (int,optional): minimum number of blocks needed for meaningful statistics, default is 4.
  Returns:
    np.array: re-blocked trace.
  """
  nblock= len(trace)//block_size
  nkeep = nblock*block_size
  if (nblock<min_nblock):
    raise RuntimeError('only %d blocks left after reblock'%nblock)
  # end if
  blocked_trace = trace[:nkeep].reshape(nblock,block_size,*trace.shape[1:])
  return np.mean(blocked_trace,axis=1)
# end def

def reblock_scalar_df(df,block_size,min_nblock=4):
  """ create a re-blocked scalar dataframe from a current scalar dataframe
   see reblock for details
  """
  return pd.DataFrame(
    reblock(df.values,block_size,min_nblock=min_nblock)
    ,columns=df.columns
  )
# end def

def ts_extrap(calc_df,issl,new_series,tname='timestep',ename='LocalEnergy',series_name='series',order=1):
  """ extrapolate dmc energy to zero time-step limit
  Args:
    calc_df (pd.DataFrame): must contain columns [tname,ename,series_name]
    issl (list): list of DMC series index to use in fit
  Returns:
    pd.Series: an entry copied from the smallest time-step DMC entry, then edited with extrapolated energy and corresponding info; series is set to -1
  """
  import scipy.optimize as op
  sel  = calc_df[series_name].apply(lambda x:x in issl)
  nfound = len(calc_df.loc[sel])
  if nfound != len(issl):
    raise RuntimeError('found %d series, when %d are requested' % (nfound,len(issl)) )
  # end if
  # !!!! need to check that the selected runs are actually DMC !
  myx  = calc_df.loc[sel,tname]
  myym = calc_df.loc[sel,ename+'_mean']
  myye = calc_df.loc[sel,ename+'_error']

  if order != 1:
    raise NotImplementedError('only linear extrapolation; order=%d not supported'%order)
  # end if
  model = lambda x,a,b:a*x+b
  popt,pcov = op.curve_fit(model,myx,myym,sigma=myye,absolute_sigma=True)
  perr = np.sqrt(np.diag(pcov))
  # return popt,perr to check fit

  y0m = popt[1]
  y0e = perr[1]

  entry = calc_df.loc[calc_df[tname]==min(myx)].copy()
  entry[tname] = 0
  entry[ename+'_mean']  = y0m
  entry[ename+'_error'] = y0e
  entry[series_name] = new_series
  return entry
# end def

def mix_est_correction(mydf,names,vseries,dseries,series_name='series',group_name='group',kind='linear',drop_missing_twists=False):
  """ extrapolate dmc energy to zero time-step limit
  Args:
    mydf (pd.DataFrame): dataframe of VMC and DMC mixed estimators
    names (list): list of DMC mixed estimators names to extrapolate
    vseries (int): VMC series id
    dseries (int): DMC series id
    series_name (str,optional): column name identifying the series
    kind (str,optinoal): extrapolation kind, must be either 'linear' or 'log'
  Returns:
    pd.Series: an entry copied from the smallest time-step DMC entry, then edited with extrapolated pure estimators. !!!! Series index is not changed!
  """
  vsel = mydf[series_name]==vseries # vmc
  msel = mydf[series_name]==dseries # mixed estimator

  # make sure the groups (twists) are aligned!!!!
  vgroup = set(mydf.loc[vsel,group_name].values)
  dgroup = set(mydf.loc[msel,group_name].values)

  missing_twists = (dgroup-vgroup).union(vgroup-dgroup)
  nmiss = len(missing_twists)
  if (nmiss>0):
    if (not drop_missing_twists):
      raise RuntimeError('twists %s incomplete, set drop_missing_twists to ignore'%' '.join([str(t) for t in missing_twists]))
    else: # drop missing twists
      good_twist = mydf.group.apply(lambda x:x not in missing_twists)
      vsel = vsel & good_twist
      msel = msel & good_twist
    # end if
  # end if

  # get values and errors
  mnames = [name+'_mean' for name in names]
  enames = [name+'_error' for name in names]
  vym = mydf.loc[vsel,mnames].values
  vye = mydf.loc[vsel,enames].values
  mym = mydf.loc[msel,mnames].values
  mye = mydf.loc[msel,enames].values

  # perform extrapolation
  if kind == 'linear':
    dym = 2.*mym - vym
    dye = np.sqrt(4.*mye**2.+vye**2.)
  elif kind == 'log':
    # extrapolate mean
    lnmym = np.log(mym)
    lnvym = np.log(vym)
    lndym = 2*lnmym-lnvym
    dym = np.exp(lndym)

    # propagate error
    lnmye = np.log(mye)
    lnvye = np.log(vye)
    lndye = np.sqrt(4.*lnmye**2.+lnvye**2.)
    dye = dym*lndye
  else:
    raise RuntimeError('unknown mixed estimator extrapolation kind = %s'%kind)
  # end if

  # store in new data frame
  puredf = mydf.loc[msel].copy()
  puredf[mnames] = dym
  return puredf
# end def