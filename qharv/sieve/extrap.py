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
