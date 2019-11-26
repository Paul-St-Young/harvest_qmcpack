# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to further process mean dataframes.
#  Mostly built around pandas's API.
#
# note: mean dataframes (mdf) are dataframes having the index of columns
#  structured as those returned by scalar_df.mean_error_scalar_df.
import numpy as np
import pandas as pd

def xyye(df, xname, yname, sel=None, xerr=False, yerr=True, sort=False):
  """Get x vs. y data from a mean data frame.

  Args:
    df (mdf): mean dataframe
    xname (str): name of x variable
    yname (str): name of y variable
    sel (np.array, optional): boolean selector for subset, default is all
    xerr (bool, optional): x variable has statistical error, default False
    yerr (bool, optional): y variable has statistical error, default True
    sort (bool, optional): sort x
  Return:
    (x, ym, ye) OR (xm, xe, ym, ye) if xerr=True
  Examples:
    >>> xyye(df, 'rs', 'LocalEnergy')
    >>> xyye(df, 'Pressure', 'LocalEnergy', xerr=True)
  """
  if sel is None:
    sel = np.ones(len(df), dtype=bool)
  xe = None
  if xerr:
    xmn = '%s_mean' % xname
    xen = '%s_error' % xname
    xm = df.loc[sel, xmn].values
    xe = df.loc[sel, xen].values
  else:
    xm = df.loc[sel, xname].values
  ye = None
  if yerr:
    ymn = '%s_mean' % yname
    yen = '%s_error' % yname
    ym = df.loc[sel, ymn].values
    ye = df.loc[sel, yen].values
  else:
    ym = df.loc[sel, yname].values
  rets = (xm, xe, ym, ye)
  idx = np.arange(len(xm))
  if sort:
    idx = np.argsort(xm)
  return [ret[idx] for ret in rets if ret is not None]

def dfme(df, labels, cols):
  """ Average scalar quantities over a set of calculations.
   This is a more intentional version of twist_average_mean_df, where
    the user specifies the groups (labels) to average over and the
    quantities to be averaged (cols). Only one assumption is made
    about content of the dataframe (col_mean, col_error).

  Args:
    df (pd.DataFrame): a mean dataframe containing labels+col_mean+col_error
    labels (list): a list of group names, e.g. ['rs', 'temperature']
    cols (list): a list of column names, e.g. ['E_tot', 'KE_tot']
  Return:
    pd.DataFrame: averaged database
  """
  mcols = ['%s_mean' % col for col in cols]
  ecols = ['%s_error' % col for col in cols]
  def sqavg(x):
    return np.sum(x**2)**0.5/len(x)
  em = df.groupby(labels)[mcols].mean()
  ee = df.groupby(labels)[ecols].apply(sqavg)
  df1 = pd.concat([em, ee], axis=1)
  return df1

def twist_average_mean_df(df0, drop_null=False):
  ''' Average scalar quantities over a set of calculations. The intented
   application is to average over a uniform grid of twist calculations.

  Args:
   df0 (pd.DataFrame): a mean dataframe containing columns
    ['path', 'fdat', '*_mean', '*_error'].
    fdat is the scalar.dat filename. It provides 'series' and 'group' indices.
   drop_null (bool, optional): drop rows that contain any nan. These rows
    should be processed at a higher level (e.g. catach the RuntimeError).
  Returns:
    pd.DataFrame: df1, a mean dataframe containing one entry for each series.
     df1 is structually identical to a mean dataframe of a single twist.
  '''
  # decide what to do with nans
  bad_sel = df0.isnull().any(axis=1)
  bad_entries = df0.loc[bad_sel]
  nbad = len(bad_entries)
  if (nbad > 0) and (not drop_null):
    msg = '%d bad runs found in scalar_df' % nbad
    msg += ', set drop_null to drop bad data.'
    raise RuntimeError(msg)
  if (drop_null):
    df0.drop(bad_entries.index, inplace=True)

  def sq_avg(trace):  # not a well-written function ...
    """ average error """
    return np.sqrt(np.sum(trace**2.))/len(trace)
  from qharv.reel import mole
  meta = df0['fdat'].apply(mole.interpret_qmcpack_fname).apply(pd.Series)
  df = pd.concat([meta, df0], axis=1)

  # average over twists
  cols = df.columns
  mcol = [col for col in cols if col.endswith('_mean')]
  ecol = [col for col in cols if col.endswith('_error')]
  rcol = [col for col in cols if (not col == 'series') and
          (not col.endswith('_mean')) and (not col.endswith('_error'))]
  groups = df.groupby('series')
  mdf = groups[mcol].apply(np.mean)
  edf = groups[ecol].apply(sq_avg)
  rdf = groups[rcol].apply(lambda x: x.drop(
   ['fdat', 'group'], axis=1).drop_duplicates().squeeze())

  df1 = rdf.join(mdf).join(edf).reset_index().sort_values('series')
  return df1
