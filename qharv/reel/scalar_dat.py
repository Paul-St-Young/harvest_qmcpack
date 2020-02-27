# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse scalar table output. Mostly built around pandas's API.

import numpy as np
import pandas as pd

def get_string_io(text):
  """Obtain StringIO object from text
   compatible with Python 2 and 3

  Args:
    text (str): text to parse
  Return:
    StringIO: file-like object
  """
  import sys
  if sys.version_info[0] < 3:
    from StringIO import StringIO
    fp = StringIO(text)
  else:
    from io import StringIO
    try:
      fp = StringIO(text.decode())
    except:
      fp = StringIO(text)
  return fp

def find_header_lines(text):
  """Find line numbers of all headers

  Args:
    text (str): text to parse
  Return:
    list: a list of integer line numbers
  """
  def is_float(s):
    try:
      float(s)
      return True
    except ValueError:
      return False
  fp = get_string_io(text)
  first_str = np.array(
    [is_float(line.split()[0]) for line in fp], dtype=bool)
  fp.close()
  idxl = np.where(~first_str)[0]
  return idxl.tolist()

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
  fp = get_string_io(text)
  # try to read header line
  header = fp.readline()
  fp.seek(0)
  # read data
  sep = r'\s+'
  if header.startswith('#'):
    df = pd.read_csv(fp, sep=sep)
    # remove first column name '#'
    columns = df.columns
    df.drop(columns[-1], axis=1, inplace=True)
    df.columns = columns[1:]
    # calculate local energy variance if possible (QMCPACK specific)
    if ('LocalEnergy' in columns) and ('LocalEnergy_sq' in columns):
      df['Variance'] = df['LocalEnergy_sq']-df['LocalEnergy']**2.
  else:
    df = pd.read_csv(fp, sep=sep, header=None)
  fp.close()
  # column labels should be strings
  df.columns = map(str, df.columns)
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

  A line is a header if its first column cannot be converted to a float.
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
  with open(dat_fname, 'r') as f:
    text = f.read()
  idxl = find_header_lines(text)
  if len(idxl) == 0:  # no header
    return [parse(text)]
  idxl.append(-1)
  # now read data and use headers to label columns
  from qharv.reel import ascii_out
  mm = ascii_out.read(dat_fname)
  dfl = []
  for bidx, eidx in zip(idxl[:-1], idxl[1:]):
    mm.seek(bidx)
    header = mm.readline()
    columns = header.replace('#', '').split()
    # read content
    text1 = mm[mm.tell():eidx]
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

def next_pow_two(n):
  """Returns the next power of two greater than or equal to `n`
  stolen from dfm/emcee autocorr.py

  Args:
    n (int): lower bound
  Return:
    n2 (int): next power of two
  Example:
    >>> next_pow_two(1000)
    >>> 1024
  """
  i = 1
  while i < n:
    i = i << 1
  return i

def acf1d(x):
  """Estimate the normalized autocorrelation function of a 1-D series
  stolen from dfm/emcee autocorr.py

  Args:
    x (np.array): The series as a 1-D numpy array.
  Returns:
    np.array: The autocorrelation function of the time series.
  """
  x = np.atleast_1d(x)
  if len(x.shape) != 1:
    msg = "invalid dimensions for 1D autocorrelation function"
    raise ValueError(msg)
  n = next_pow_two(len(x))

  # Compute the FFT and then (from that) the auto-correlation function
  f = np.fft.fft(x - np.mean(x), n=2*n)
  acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
  # normalize
  acf0 = acf[0]
  if not np.isclose(acf0, 0):
    acf /= acf[0]
  return acf

def corr(trace):
  """ calculate the autocorrelation of a trace of scalar data

  correlation time is defined as the integral of the auto-correlation
   function from t=0 to when the function first reaches 0.

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
  Return:
    float: correlation_time, the autocorrelation time of this trace of scalars
  """
  acf = acf1d(trace)
  # find first zero crossing
  sel = acf < 0
  neg_itau = np.arange(len(acf))[sel]
  if len(neg_itau) == 0:
    return np.inf
  itau = neg_itau[0]
  correlation_time = 1.0 + 2.0*np.sum(acf[:itau])
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
