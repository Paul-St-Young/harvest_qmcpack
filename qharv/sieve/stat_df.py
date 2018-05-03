# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to roughly process stat Dataframes.
# Mostly built around pandas's API.
import numpy as np
import pandas as pd


def mix_extrap_gofr(y0m, y0e, y1m, y1e, ythres=1e-2):
  """ use VMC g(r) to correct mixed-estimator DMC g(r)
  Args:
    y0m (np.array): mean  of VMC g(r)
    y0e (np.array): error of VMC g(r)
    y1m (np.array): mean  of DMC g(r)
    y1e (np.array): error of DMC g(r)
    ythres (float, optional): VMC g(r) < ythres will not be used
     to avoid numerical instability, default 0.01
  Returns:
    tuple: (y2m,y2e) extrapolated g(r) mean and error
  """

  # inputs must all be numpy arrays
  for arr in [y0m, y0e, y1m, y1e]:
    if type(arr) is not np.ndarray:
      raise TypeError()

  # select r @ non-zero portion of g(r)
  sel = (y0m > ythres) & (y1m > ythres)

  # extrapolate log
  lny0m = np.log(y0m[sel])
  lny1m = np.log(y1m[sel])
  lny0e = y0e[sel]/y0m[sel]
  lny1e = y1e[sel]/y1m[sel]
  lny2m = 2*lny1m-lny0m
  lny2e = np.sqrt(4*lny1e**2.+lny0e**2.)

  # create new entry of g(r) mean and error
  y2m = y1m.copy()
  y2e = y1e.copy()
  y2m[sel] = np.exp(lny2m)
  y2e[sel] = y2m[sel] * lny2e

  return y2m, y2e


def pure_gofr(entry, iss0, iss1, iss2, yname
              , series_name='series'):

  # select VMC and DMC entries
  sel0 = entry[series_name] == iss0
  sel1 = entry[series_name] == iss1
  assert len(entry.loc[sel0]) == 1
  assert len(entry.loc[sel1]) == 1

  # perform extrapolation
  ymean_name = yname + '_mean'
  yerror_name = yname + '_error'
  y0m = np.array(entry.loc[sel0, ymean_name].squeeze())
  y0e = np.array(entry.loc[sel0, yerror_name].squeeze())
  y1m = np.array(entry.loc[sel1, ymean_name].squeeze())
  y1e = np.array(entry.loc[sel1, yerror_name].squeeze())
  y2m, y2e = mix_extrap_gofr(y0m, y0e, y1m, y1e)

  # create new entry
  entry2 = entry.loc[sel1].copy()
  entry2[ymean_name]  = [y2m.tolist()]
  entry2[yerror_name] = [y2e.tolist()]
  entry2[series_name] = iss2
  return entry2
