# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to roughly process scalar Dataframes.
# Mostly built around pandas's API.
#
#  note: A scalar dataframe (scalar_df) is expected to contain the raw data,
# i.e. block-resolved expectation values, of a SINGLE calculation.
# If multiple runs are collected in the same dataframe, label by ['path'
# , 'fdat'] and use groupby before applying the functions in this script.
import numpy as np
import pandas as pd
from qharv.reel.scalar_dat import error

def merge_list(dfl, labels):
  """Merge a list of DataFrames sharing common labels

  Args:
    dfl (list): a list of pd.DataFrame objects
    labels (list): a list of column labels
  Return:
    pd.DataFrame: merged df
  """
  import sys
  if sys.version[0] != "2":
    from functools import reduce
  df = reduce(lambda df1, df2: pd.merge(df1, df2, on=labels), dfl)
  return df

def mean_error_scalar_df(df, nequil=0):
  """ get mean and average from a dataframe of raw scalar data (per-block)
   take dataframe having columns ['LocalEnergy','Variance',...] to a
   dataframe having columns ['LocalEnergy_mean','LocalEnergy_error',...]

   Args:
    df (pd.DataFrame): raw scalar dataframe, presumable generated using
     qharv.scalar_dat.parse with extra labels columns added to identify
     the different runs.
    nequil (int, optional): number of equilibration blocks to throw out
     for each run, default 0 (keep all data).
   Returns:
    pd.DataFrame: mean_error dataframe
  """
  from qharv.sieve import mean_df
  if nequil > 0:
    if 'index' not in df.columns:
      msg = 'time series must be indexed to drop equilibration,'
      msg += ' please add "index" to DataFrame column.'
      raise RuntimeError(msg)
    sel = df['index'] >= nequil  # zero indexing
    mydf = df.loc[sel]
  else:  # allow equilibration to be dropped outside of this function
    mydf = df
  return mean_df.create(mydf)

def reblock(trace, block_size, min_nblock=4, with_sigma=False):
  """ block scalar trace to remove autocorrelation;
  see usage example in reblock_scalar_df

  Args:
    trace (np.array): a trace of scalars, may have multiple columns
      !!!! assuming leading dimension is the number of current blocks.
    block_size (int): size of block in units of current block.
    min_nblock (int,optional): minimum number of blocks needed for
      meaningful statistics, default is 4.
  Returns:
    np.array: re-blocked trace.
  """
  nblock= len(trace)//block_size
  nkeep = nblock*block_size
  if (nblock < min_nblock):
    raise RuntimeError('only %d blocks left after reblock' % nblock)
  # end if
  blocked_trace = trace[:nkeep].reshape(nblock, block_size, *trace.shape[1:])
  ret = np.mean(blocked_trace, axis=1)
  if with_sigma:
    ret = (ret, np.std(blocked_trace, ddof=1, axis=1))
  return ret

def reblock_scalar_df(df, block_size, min_nblock=4):
  """ create a re-blocked scalar dataframe from a current scalar dataframe
   see reblock for details
  """
  return pd.DataFrame(
    reblock(df.values, block_size, min_nblock=min_nblock),
    columns=df.columns
  )

def poly_extrap_to_x0(myx, myym, myye, order, return_fit=False):
  """ fit 1D data to 1D polynomial and extrpolate to x=0

  The fit proceeds in two steps. The first polyfit does not take error into
  account. It estimates the extrapolated value, which is then used to setup
  a trust region (bounds). Using the trust region, curve_fit can robustly
  estimate the error of the extrapolation.

  Args:
    myx (np.array): x values
    myym (np.array): y values
    myye (np.array): y errors (1 sigma)
    order (int): order of 1D polynomial
    return_fit (bool, optional): if true, then return fit paramters
  Return:
    2-tuple: floats (y0m, y0e), y mean and error at x=0
  """
  import scipy.optimize as op

  if order != 1:
    raise NotImplementedError('order=%d not supported' % order)
  # keep target as zeroth parameter
  model = lambda x, a, b: a+b*x

  # setup trust region using 10*sigma around naive extrapolation

  #  first do a fit without error
  popt0 = np.polyfit(myx, myym, order)
  val0 = np.poly1d(popt0)(0)

  #  then use rough fit to setup trust region
  sig0 = max(myye)  # extrapolated error should be larger than all data
  nsig = 10  # !!!! hard-code 10 sigma
  lbounds = [-np.inf for i in range(order+1)]
  ubounds = [ np.inf for i in range(order+1)]
  lbounds[0] = val0 - nsig*sig0
  ubounds[0] = val0 + nsig*sig0
  bounds = (lbounds, ubounds)

  # finally fit using error and trust region
  popt, pcov = op.curve_fit(model, myx, myym,
    sigma=myye, absolute_sigma=True, bounds=bounds, method='trf')
  perr = np.sqrt(np.diag(pcov))
  # return popt,perr to check fit

  y0m = popt[0]
  y0e = perr[0]
  ret = (y0m, y0e)
  if return_fit:
    ret = (y0m, y0e, popt, pcov)
  return ret

def ts_extrap_obs(calc_df, sel, tname, obs, order=1):
  """ extrapolate a single dmc observable to zero time-step limit

  Args:
    calc_df (pd.DataFrame): must contain columns [tname, obs_mean, obs_error]
    sel (np.array): boolean selector array
    tname (str): timestep column name, e.g. 'timestep'
    obs (str): observable column name, e.g. 'LocalEnergy'
  Return:
    tuple: (myx, y0m, y0e) of type (list, float, float) containing
    (timesteps, t=0 value, t=0 error)
  """

  # !!!! need to check that the selected runs are actually DMC !
  myx  = np.array(calc_df.loc[sel, tname].values)
  myym = np.array(calc_df.loc[sel, obs+'_mean'].values)
  myye = np.array(calc_df.loc[sel, obs+'_error'].values)

  y0m, y0e = poly_extrap_to_x0(myx, myym, myye, order)

  return myx, y0m, y0e

def ts_extrap(calc_df, issl, obsl,
  tname='timestep', series_name='series', **kwargs):
  """ extrapolate all dmc observables to zero time-step limit

  Args:
    calc_df (pd.DataFrame): must contain columns [tname, series_name]
    issl (list): list of DMC series index to use in fit
    obsl (list): a list of observable names to extrapolate
  Return:
    pd.Series: an entry copied from the smallest time-step DMC entry,
    then edited with extrapolated energy and corresponding info
    !!!! series number is unchanged
  """

  sel  = calc_df[series_name].apply(lambda x: x in issl)
  nfound = len(calc_df.loc[sel])
  if nfound != len(issl):
    msg = 'found %d series, when %d are requested' % (nfound, len(issl))
    raise RuntimeError(msg)

  # copy smallest timestep DMC entry
  myx  = calc_df.loc[sel, tname]
  entry = calc_df.loc[calc_df[tname] == min(myx)].copy()

  # fill entry with new data
  entry[tname] = 0

  for obs in obsl:
    myx0, y0m, y0e = ts_extrap_obs(calc_df, sel,
      tname, obs, **kwargs)
    entry['%s_mean' % obs]  = y0m
    entry['%s_error' % obs] = y0e
  return entry

def mix_est_correction(mydf, vseries, dseries, namesm,
  series_name='series', group_name='group', kind='linear',
  drop_missing_twists=False):
  """ extrapolate dmc energy to zero time-step limit
  Args:
    mydf (pd.DataFrame): dataframe of VMC and DMC mixed estimators
    vseries (int): VMC series id
    dseries (int): DMC series id
    names (list): list of DMC mixed estimators names to extrapolate
    series_name (str,optional): column name identifying the series
    kind (str,optinoal): extrapolation kind, must be either 'linear' or 'log'
  Returns:
    pd.Series: an entry copied from the smallest time-step DMC entry,
      then edited with extrapolated pure estimators.
      !!!! Series index is not changed!
  """
  vsel = mydf[series_name] == vseries  # vmc
  msel = mydf[series_name] == dseries  # mixed estimator

  # make sure the groups (twists) are aligned!!!!
  vgroup = set(mydf.loc[vsel, group_name].values)
  dgroup = set(mydf.loc[msel, group_name].values)

  missing_twists = (dgroup-vgroup).union(vgroup-dgroup)
  nmiss = len(missing_twists)
  if (nmiss > 0):
    if (not drop_missing_twists):
      msg = 'twists ' + ' '.join([str(t) for t in missing_twists])
      msg += ' incomplete, set drop_missing_twists to ignore'
      raise RuntimeError(msg)
    else:  # drop missing twists
      good_twist = mydf.group.apply(lambda x: x not in missing_twists)
      vsel = vsel & good_twist
      msel = msel & good_twist
    # end if
  # end if

  # get values and errors
  mnames = [name+'_mean' for name in names]
  enames = [name+'_error' for name in names]
  vym = mydf.loc[vsel, mnames].values
  vye = mydf.loc[vsel, enames].values
  mym = mydf.loc[msel, mnames].values
  mye = mydf.loc[msel, enames].values

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
    msg = 'unknown mixed estimator extrapolation kind = %s' % kind
    raise RuntimeError(msg)
  # end if

  # store in new data frame
  puredf = mydf.loc[msel].copy()
  puredf[mnames] = dym
  return puredf
