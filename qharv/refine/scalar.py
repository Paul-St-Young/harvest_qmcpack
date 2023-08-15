# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to refine collected data to be presentable
import numpy as np
import pandas as pd
from qharv.field import sugar

def text_mean_error(ym_in, ye_in, nshow=1, mndig=8):
  """ convert data such as 1.23 +/- 0.01 to strings such as 1.23(1)

  Args:
    ym (np.array): mean
    ye (np.array): error
  Return:
    np.array: an array of strings
  """
  # handle number of list inputs
  if np.issubdtype(type(ym_in), np.number):
    ym = np.array([ym_in])
    ye = np.array([ye_in])
  else:
    ym = np.array(ym_in)
    ye = np.array(ye_in)
  # find the number of digits to print
  ndig = mndig*np.ones(len(ye), dtype=int)
  sel = abs(ye) >= 10**(-mndig)
  # last nshow digits are uncertain
  ndig[sel] = np.ceil(-np.log10(abs(ye[sel]))).astype(int)
  #  in case no floating point part (use scientific notation pls)
  sel = ndig < 0
  ndig[sel] = 0
  # print the desired number of digits
  ymt = []
  for (y, n) in zip(ym, ndig):
    fmt = '%10.'+str(n+nshow-1)+'f'
    ymt.append(fmt % y)
  # get last digit error
  ye_val = np.around(abs(ye)*10**(ndig+nshow-1))
  yet = ['('+e.astype(int).astype(str)+')' if e > 0 else '' for e in ye_val]
  # append error in parenteses
  yt = [(m+e).strip() for (m, e) in zip(ymt, yet)]
  if len(yt) == 1:
    return yt[0]
  return np.array(yt)

def mean_error_text(texts):
  """ convert strings such as 1.23(1) to data such as 1.23 +/- 0.01
  i.e. inverse of text_mean_error

  Args:
    texts (np.array): an array of texts
  Return:
    (np.array, np.array): (ym, ye), (mean, error)
  """
  def met(t):
    ymt, yet = t.split('(')
    if '.' in ymt:
      ndig = len(ymt.split('.')[1])
    else:
      ndig = 0
    ym = float(ymt.replace(' ', ''))
    ye = float(yet.replace(')', ''))
    return ym, ye*10**(-ndig)
  return np.vectorize(met)(texts)

def text_df(df, obsl, **kwargs):
  """ write a subset of df into readable text

  for each observable in obsl, there must be a mean and an error column
  associated with it. The column names must be obs+'_mean' and obs+'_error'

  Args:
    df (pd.DataFrame): database of Monte-Carlo data with *_mean and *_error
    obsl (array-like): list of observable names
  """
  tdata = {}
  for obs in obsl:
    mcol = obs+'_mean'
    ecol = obs+'_error'
    ym = df[mcol].values
    ye = df[ecol].values
    yt = text_mean_error(ym, ye, **kwargs)
    tdata[obs] = yt

  tdf = pd.DataFrame(tdata)
  return tdf

def text_df_obs_exobs(df, obsl, exobsl, **kwargs):
  """ construct text dataframe

  assume obsl have associated _mean and _error columns.

  Args:
    df (pd.DataFrame): scalar database
    obsl (list): a list of observable names, each with _mean and _error
    exobsl (list): a list of exact observable names
  Return:
    pd.DataFrame: text database
  """
  tdf = text_df(df, obsl, **kwargs)
  for col in exobsl:
    tdf[col] = df[col].values
  return tdf[exobsl+obsl]

@sugar.cache
def write_latex_table(table_tex, tdf, **kwargs):
  """ write LaTeX table using a chunk of database df
  will throw exception is table_tex already exists on disk

  Args:
    table_tex (str): latex file to hold table
    tdf (pd.DataFrame): text database
  """
  if 'escape' not in kwargs:
    kwargs['escape'] = False  # allow $ in column names
  if 'index' not in kwargs:
    kwargs['index'] = False
  tdf.to_latex(table_tex, **kwargs)
