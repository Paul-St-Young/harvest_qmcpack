# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to further process mean dataframes.
#  Mostly built around pandas's API.
#
# note: mean dataframes (mdf) are dataframes having the index of columns
#  structured as those returned by scalar_df.mean_error_scalar_df.
import numpy as np
import pandas as pd

def categorize_columns(cols):
  """Categorize the column names of a mean dataframe.

  Args:
    cols (list): a list of column names
  Return:
    (list, list, list): (excol, mcol, ecol)
      excol are columns of exact values with no errorbar (possibly labels)
      mcols are mean columns
      ecols are error columns
  Examples:
    >>> rcol, mcol, ecol = categorize_columns(mdf.columns)
    >>> xyye(df, 'Pressure', 'LocalEnergy', xerr=True)
  """
  mcol = [col for col in cols if col.endswith('_mean')]
  ecol = [col for col in cols if col.endswith('_error')]
  rcol = [col for col in cols if
          (not col.endswith('_mean')) and (not col.endswith('_error'))]
  return rcol, mcol, ecol

def xyye(df, xname, yname, sel=None, xerr=False, yerr=True, sort=False):
  """Get x vs. y data from a mean data frame.

  Args:
    df (pd.DataFrame): mean dataframe
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

def taw(ym, ye, weights):
  """ twist average with weights """
  wtot = weights.sum()
  try:
    aym = np.dot(ym, weights)/wtot
    aye = np.dot(ye**2, weights**2)**0.5/wtot
  except ValueError as err:
    if 'not aligned' not in str(err):
      raise err
    aym = (weights[:, np.newaxis]*ym).sum(axis=0)/wtot
    aye = (weights[:, np.newaxis]**2*ye**2).sum(axis=0)**0.5/wtot
  return aym, aye

def dfme(df, cols, no_error=False, weight_name=None):
  """ Average scalar quantities over a set of calculations.

  Args:
    df (pd.DataFrame): a mean dataframe containing labels+col_mean+col_error
    cols (list): a list of column names, e.g. ['E_tot', 'KE_tot']
    weight_name (str, optional): name of weight column, default None, i.e.
     every entry has the same weight
  Return:
    pd.DataFrame: averaged database
  """
  # goal: build pd.Series containing average
  entry = {}
  # extract weights
  if weight_name is None:
    wts = np.ones(len(df))
  else:
    wts = df[weight_name].values
  # average with weights
  if no_error:
    mcols = cols
    datm = df[cols].values
    ym, ye_junk = taw(datm, datm, wts)
  else:
    mcols = ['%s_mean' % col for col in cols]
    ecols = ['%s_error' % col for col in cols]
    datm = df[mcols].values
    date = df[ecols].values
    ym, ye = taw(datm, date, wts)
    for col, y1 in zip(ecols, ye):
      entry[col] = y1
  for col, y1 in zip(mcols, ym):
    entry[col] = y1
  return pd.Series(entry)
