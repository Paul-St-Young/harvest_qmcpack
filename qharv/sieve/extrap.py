# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to perform extrapolations with dataframes.
import numpy as np
import pandas as pd

def polyextrap(x, ym, ye=None, order=1, xtarget=0, return_fit=False):
  """Extrapolate y to x->0 limit using a polynomial fit.

  Inputs:
    x (np.array): shape (npt,), x values
    ym (np.array): shape (npt,), y values
    ye (np.array, optional): shape (npt,), y errors, default 0
    order (int, optional): polynomial order, default 1
    xtarget (x.dtype): target value of extrapolation, default 0
    return_fit (bool, optional): return fit parameters, default False
  """
  from scipy.optimize import curve_fit
  # first use deterministic fit to estimate parameters
  popt0 = np.polyfit(x, ym, order)
  if ye is None:
    y0 = np.poly1d(popt0)(xtarget)
    if return_fit:
      return (y0, popt0)
    return y0
  # estimate fit error
  popt, pcov = curve_fit(lambda x, *p: np.poly1d(p)(x),
    x, ym, sigma=ye, absolute_sigma=True, p0=popt0)
  perr = np.sqrt(np.diag(pcov))
  y0m = np.poly1d(popt)(xtarget)
  if xtarget == 0:
    y0e = perr[-1]
  else:
    msg = 'general error estimate'
    raise NotImplementedError(msg)
  y0 = (y0m, y0e)
  ret = y0
  if return_fit:
    ret = (y0, popt, perr)
  return ret

def mix_extrap(df, yname):
  assert df.index.is_unique
  # extract VMC value
  sel0 = df.method == 'vmc'
  assert len(df.loc[sel0]) == 1
  ymn = '%s_mean' % yname
  yen = '%s_error' % yname
  y0m = df.loc[sel0, ymn].values.squeeze()
  y0e = df.loc[sel0, yen].values.squeeze()
  # calculate 2*DMC-VMC
  sel1 = df.method == 'dmc'
  dm = 2*df.loc[sel1, ymn]-y0m
  de = ((4*df.loc[sel1, yen])**2+y0e**2)**0.5
  df1 = pd.concat([dm, de], axis=1)
  return df1

def mix_estimator(df, ynames):
  labels = ['path', 'id']
  sufs = ['_mean', '_error']
  for yname in ynames:
    mydf = df.groupby(labels).apply(
      mix_extrap, yname).reset_index()
    # find entries in original data frame
    idx = mydf['level_2'].values
    for suf in sufs:
      col = '%s%s' % (yname, suf)
      icol = df.columns.get_loc(col)
      # set entry
      df.iloc[idx, icol] = mydf[col].values

def ts_extrap(df, yname, xname='timestep', plot=False):
  from qharv.sieve import mean_df
  x, ym, ye = mean_df.xyye(df, xname, yname)
  y0fit = polyextrap(x, ym, ye, return_fit=plot)
  if plot:
    y0, popt, perr = y0fit
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    line = ax.errorbar(x, ym, ye, ls='', marker='.')
    finex = np.linspace(1e-3, x.max())
    finey = np.poly1d(popt)(finex)
    ax.plot(finex, finey, c=line[0].get_color())
    plt.show()
  else:
    y0 = y0fit
  y0m, y0e = y0
  ret = pd.Series({
    '%s_mean' % yname: y0m,
    '%s_error' % yname: y0e,
  })
  return ret

def timestep(df, labels, ynames, xname='timestep', plot=False):
  sufs = ['_mean', '_error']
  # keep one entry per group
  df1 = df.groupby(labels).first()
  df1.drop(columns=xname, inplace=True)
  # replace targeted columns with extrapolated data
  for yname in ynames:
    # extrapolate
    mydf = df.groupby(labels).apply(
      ts_extrap, yname, xname=xname, plot=plot)
    # replace
    for suf in sufs:
      col = '%s%s' % (yname, suf)
      df1[col] = mydf[col]
  return df1.reset_index()
