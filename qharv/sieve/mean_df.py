# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to further process mean dataframes.
#  Mostly built around pandas's API.
#
# note: mean dataframes (mdf) are dataframes, with the specific structure
#  as defined in mean_df.create. Observable "yname" with statistical error
#  is stored in columns "${yname}_mean" and "${yname}_error". "yname" with
#  no error bar is simply kept in column "$yname".
#  Columns can be categorized using mean_df.categorize_columns, and
#   extracted using mean_df.xyye.
import numpy as np
import pandas as pd

# ======================== level 1: basic I/O =========================
def create(mydf):
  # create pd.Series of mean
  msr = mydf.apply(np.mean)
  if 'index' in msr:
    msr.drop('index', inplace=True)

  # create pd.Series of errror
  try:  # use fortran library to recalculate kappa if compiled
    from qharv.reel.forlib.stats import error
  except ImportError as err:
    msg = str(err)
    msg += '\n  Please compile qharv.reel.forlib.stats using f2py.'
    raise ImportError(msg)
  efunc = error  # override error function here

  esr = mydf.apply(  # error cannot be directly applied to matrix yet
    lambda x: float(np.apply_along_axis(efunc, 0, x))
  )
  if 'index' in esr:
    esr.drop('index', inplace=True)

  # create _mean _error dataframe
  df1 = msr.to_frame().T
  df2 = esr.to_frame().T
  jdf = df1.join(df2, lsuffix='_mean', rsuffix='_error')
  return jdf

def categorize_columns(cols, nosuf=False, msuffix='_mean', esuffix='_error'):
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
    >>> rcol, names = categorize_columns(mdf.columns, nosuf=True)
    >>> np.allclose([name+'_mean' for name in names], mcol)
    True
  """
  mcol = [col for col in cols if col.endswith(msuffix)]
  ecol = [col for col in cols if col.endswith(esuffix)]
  rcol = [col for col in cols if
          (not col.endswith(msuffix)) and (not col.endswith(esuffix))]
  if nosuf:
    names = [col.replace(msuffix, '') for col in mcol]
    return rcol, names
  return rcol, mcol, ecol

def xyye(df, xname, yname, sel=None, xerr=False, yerr=True, sort=False, msuffix='_mean', esuffix='_error'):
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
    (x, y)           if xerr=False, yerr=False
    (x, ym, ye)      if xerr=False, yerr=True
    (xm, xe, ym, ye) if xerr=True,  yerr=True
  Examples:
    >>> xyye(df, 'rs', 'LocalEnergy')
    >>> xyye(df, 'Pressure', 'LocalEnergy', xerr=True)
  """
  if sel is None:
    sel = np.ones(len(df), dtype=bool)
  xe = None
  if xerr:
    xmn = '%s%s' % (xname, msuffix)
    xen = '%s%s' % (xname, esuffix)
    xm = df.loc[sel, xmn].values
    xe = df.loc[sel, xen].values
  else:
    xm = df.loc[sel, xname].values
  ye = None
  if yerr:
    ymn = '%s%s' % (yname, msuffix)
    yen = '%s%s' % (yname, esuffix)
    ym = df.loc[sel, ymn].values
    ye = df.loc[sel, yen].values
  else:
    ym = df.loc[sel, yname].values
  rets = (xm, xe, ym, ye)
  idx = np.arange(len(xm))
  if sort:
    idx = np.argsort(xm)
  return [ret[idx] for ret in rets if ret is not None]

def dxyye(xy1, xy0):
  """Calculate y1-y0 at points that exist in both x1 and x0

  Args:
    xy1 (tuple): (x, y) or (x, ym, ye) or data to plot
    xy0 (tuple): (x, y) or (x, ym, ye) or reference
  Return:
    (x, y) or (x, ym, ye)
  Examples:
    >>> xy0 = xyye(df, 'rs', 'LocalEnergy', sel=df.crystal=='c2c')
    >>> xy1 = xyye(df, 'rs', 'LocalEnergy', sel=df.crystal=='cmca4')
    >>> x, ym, ye = dxyye(xy1, xy0)
  """
  if len(xy1) == 3:  # with errorbar
    x1, ym1, ye1 = xy1
    x0, ym0, ye0 = xy0
    x, sel1, sel0 = np.intersect1d(x1, x0, return_indices=True)
    ym = ym1[sel1]-ym0[sel0]
    ye = (ye1[sel1]**2+ye0[sel0]**2)**0.5
    return x, ym, ye
  elif len(xy1) == 2:  # no errorbar
    x1, y1 = xy1
    x0, y0 = xy0
    x, sel1, sel0 = np.intersect1d(x1, x0, return_indices=True)
    y = y1[sel1]-y0[sel0]
    return x, y
  else:
    msg = 'unknown case'
    raise RuntimeError(msg)

def group_min(df, labels, yname, find_max=False):
  """Select the row that minimizes column "yname" in each group.
  The groups are organized by "labels"

  Args:
    df (pd.DataFrame): must have labels + [yname] in columns
    labels (list): a list of columns to use as group labels
    yname (str): column to minimize
    find_max (bool, optional): maximize "yname", default False
  Return:
    pd.DataFrame: a subset of rows from original df
  Example:
    >>> # extract smallest timestep runs
    >>> smalldf = group_min(df, ['nelec', 'nwalker'], 'timestep')
    >>> # find ground state at each density and magnetization
    >>> gsdf = group_min(df, ['rs', 'magnetization'], 'energy')
  """
  groups = df.groupby(labels)
  if find_max:
    idx = groups.apply(lambda x: x[yname].idxmax()).values
  else:
    idx = groups.apply(lambda x: x[yname].idxmin()).values
  return df.iloc[idx]

# ======================== level 2: propagate error =======================
def resample(marr, earr, nsample, seed=None):
  shape = marr.shape
  assert np.allclose(shape, earr.shape)
  if seed is not None:
    np.random.seed(seed)
  noise = earr*np.random.randn(nsample, *shape)
  new_marr = marr[np.newaxis] + noise
  return new_marr

# ======================== level 2: twist average =========================
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

# ======================== level 3: extrap. =========================

def linex(df_in, vseries, dseries, names, labels=None,
  sorted_df=False, sort_col=None):
  """ Linearly extrapolate to 2*DMC-VMC
  hint: can also do time-step extrapolation if tau1 = 2*tau2

  Args:
    mydf (pd.DataFrame): database, must contain ['series'] +
     name_mean, name_error for name in names
    vseries (int): VMC series index
    dseries (int): DMC series index
    names (list): a list of observable names to be extrpolated
    labels (list, optinal): a list of labels columns to keep along
      observables, default None
    sorted_df (bool, optional): VMC and DMC entries in input df are aligned
    sort_col (str, optional): column used to sort entries before subtraction
  Return:
    pd.DataFrame: extrapolated entry
  """
  if (len(df_in) != 2) and (not sorted_df):  # input must be sorted in advance
    if sort_col is None:
      msg = 'must provide sort_col if input df is not sorted'
      raise RuntimeError(msg)
    else:   # sort input to align VMC and DMC data
      mydf = df_in.sort_values(sort_col)
  else:  # assume input is already sorted correctly
    mydf = df_in
  # extract data
  mcols = ['%s_mean' % col for col in names]
  ecols = ['%s_error' % col for col in names]
  vsel = mydf.series == vseries
  vmarr = mydf.loc[vsel, mcols].values.astype(float)
  vearr = mydf.loc[vsel, ecols].values.astype(float)
  dsel = mydf.series == dseries
  dmarr = mydf.loc[dsel, mcols].values.astype(float)
  dearr = mydf.loc[dsel, ecols].values.astype(float)
  # linearly extrapolate
  pmarr = 2*dmarr-vmarr
  pearr = (4*dearr**2+vearr**2)**0.5
  # add labels
  if labels is not None:
    meta = mydf.loc[dsel, labels].values
    data = np.concatenate([meta, pmarr, pearr], axis=1)
  else:
    data = np.concatenate([pmarr, pearr], axis=1)
  mycols = [] if labels is None else labels
  pdf = pd.DataFrame(data, columns=mycols+mcols+ecols)
  return pdf
