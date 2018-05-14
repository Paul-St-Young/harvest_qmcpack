# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to refine collected data to be presentable
import numpy as np
import pandas as pd


def text_mean_error(ym, ye):
  """ convert data such as 1.23 +/- 0.01 to strings such as 1.23(1)

  Args:
    ym (np.array): mean
    ye (np.array): error
  Return:
    np.array: an array of strings
  """

  # find the number of digits to print
  ndig = np.around(-np.log10(ye)).astype(int) + 1  # last digit is uncertain

  # print the desired number of digits
  ymt = [str(np.around(y, n)).zfill(n) for (y, n) in zip(ym, ndig)]

  # get last digit error
  yet = np.around(ye*10**(ndig-1)).astype(int).astype(str)

  # append error in parenteses
  yt = [m+'('+e+')' for (m, e) in zip(ymt, yet)]
  return np.array(yt)


def text_df_sel(df, sel, obsl):
  """ write a subset of df into readable text

  for each observable in obsl, there must be a mean and an error column
  associated with it. The column names must be obs+'_mean' and obs+'_error'

  Args:
    df (pd.DataFrame): database of Monte-Carlo data with *_mean and *_error
    sel (np.array): array of boolean, i.e. row selector
    obsl (array-like): list of observable names
  """
  tdata = {}
  for obs in obsl:
    mcol = obs+'_mean'
    ecol = obs+'_error'
    ym = df.loc[sel, mcol].values
    ye = df.loc[sel, ecol].values
    yt = text_mean_error(ym, ye)
    tdata[obs] = yt

  tdf = pd.DataFrame(tdata)
  return tdf
