# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse scalar table output. Mostly built around pandas's API.

import numpy as np
import pandas as pd

def parse(text):
  """ Parse text of a scalar.dat file, should be table format.

  Args:
    text (str): content of scalar.dat file
  Return:
    pd.DataFrame: table data
  Example:
    >>> with open('vmc.s001.scalar.dat', 'r') as f:
    >>>   text = f.read()
    >>>   df = parse(text)
  """
  import sys
  if sys.version_info[0] < 3:
    from StringIO import StringIO
  else:
    from io import StringIO
  lines = text.split('\n')
  # try to read header line
  header = lines[0]
  sep = r'\s+'
  # read data
  if header.startswith('#'):
    df = pd.read_csv(StringIO(text), sep=sep)
    # remove first column name '#'
    columns = df.columns
    df.drop(columns[-1], axis=1, inplace=True)
    df.columns = columns[1:]
  else:
    df = pd.read_csv(StringIO(text), sep=sep, header=None)
  return df

def read(dat_fname):
  """ Read the scalar.dat file, should be table format.
   The header line should start with '#' and contain column labels.

  Args:
    dat_fname (str): name of input file
  Return:
    pd.DataFrame: df containing the table of data
  Example:
    >>> df = read('vmc.s001.scalar.dat')
  """
  with open(dat_fname, 'r') as f:
    text = f.read()
  return parse(text)

def read_to_list(dat_fname):
  """ read scalar.dat file into a list of pandas DataFrames

  Header lines should start with '#', assumed to contain column labels.
  Many scalar.dat files can be concatenated. A list will be returned.

  Args:
    dat_fname (str): name of input file
  Return:
    list: list of df(s) containing the table(s) of data
  Example:
    >>> dfl = read_to_list('gctas.dat')
    >>> df = pd.concat(dfl).reset_index(drop=True)
  """
  # first separate out the header lines and parse them
  from qharv.reel import ascii_out
  mm = ascii_out.read(dat_fname)
  idxl = ascii_out.all_lines_with_tag(mm, '#')
  bidxl = []
  headerl = []
  for idx in idxl:
    mm.seek(idx)
    header = mm.readline()
    bidxl.append(mm.tell())
    headerl.append(header)
  idxl.append(-1)
  # now read data and use headers to label columns
  dfl = []
  for bidx, eidx, header in zip(bidxl, idxl[1:], headerl):
    columns = header.replace('#', '').split()
    text1 = mm[bidx:eidx]
    df1 = parse(text1)
    df1.columns = columns
    dfl.append(df1)
  return dfl

def write(dat_fname, df, header_pad='# ', **kwargs):
  """ write dataframe to plain text scalar table format

  Lightly wrap around pandas.to_string with defaults to index and float_format

  Args:
    dat_fname (str): output data file name
    df (pd.DataFrame): data
    header_pad (str, optional): pad beginning of header with comment
     string, default '# '
  """
  default_kws = {
    'index': False,
    'float_format': '%8.6f'
  }
  for k, v in default_kws.items():
    if k not in kwargs:
      kwargs[k] = v
  text = df.to_string(**kwargs)
  with open(dat_fname, 'w') as f:
    f.write(header_pad + text)

def corr(trace):
  """ calculate the autocorrelation of a trace of scalar data

  correlation time is defined as the integral of the auto-correlation
   function from t=0 to when the function first reaches 0.

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
  Return:
    float: correlation_time, the autocorrelation time of this trace of scalars
  """

  mu     = np.mean(trace)
  stddev = np.std(trace, ddof=1)
  if np.isclose(stddev, 0):  # easy case
    return np.inf  # infinite correlation for constant trace
  correlation_time = 0.
  for k in range(1, len(trace)):
    # calculate auto_correlation
    auto_correlation = 0.0
    num = len(trace)-k
    auto_correlation = np.dot(trace[:num]-mu, trace[k:]-mu)
    auto_correlation *= 1.0/(num*stddev**2)
    if auto_correlation > 0:
      correlation_time += auto_correlation
    else:
      break
  correlation_time = 1.0 + 2.0*correlation_time
  return correlation_time

def error(trace, kappa=None):
  """ calculate the error of a trace of scalar data

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
    kappa (float,optional): auto-correlation time, default is to re-calculate
  Return:
    float: stderr, the error of the mean of this trace of scalars
  """
  stddev = np.std(trace, ddof=1)
  if np.isclose(stddev, 0):  # easy case
    return 0.0  # no error for constant trace
  if kappa is None:  # no call to corr
    kappa = corr(trace)
  neffective = np.sqrt(len(trace)/kappa)
  err = stddev/neffective
  return err

def single_column(df, column, nequil):
  """ calculate mean and error of a column

    nequil blocks of data are thrown out; autocorrelation time is taken into
  account when calculating error. The equilibrated data is assumed to have
  Gaussian distribution. Error is calculated for one standard deviation
  (1-sigma error).

  Args:
    df (pd.DataFrame): table of data (e.g. from scalar_dat.parse)
    column (str): name of column
    nequil (int): number of equilibration blocks
  Return:
    (float,float,float): (ymean,yerr,ycorr), where ymean is the mean of column
     , yerr is the 1-sigma error of column, and ycorr is the autocorrelation
  """

  myy = df[column].values[nequil:]

  ymean = np.mean(myy)
  ycorr = corr(myy)
  yerr  = error(myy, ycorr)
  return ymean, yerr, ycorr
