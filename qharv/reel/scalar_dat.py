# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse scalar table output. Mostly built around pandas's API.

import numpy as np
import pandas as pd

def read(dat_fname, **kwargs):
  """Read the scalar.dat file, should be table format.
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
  return parse(text, **kwargs)

def write(dat_fname, df, header_pad='# ', **kwargs):
  """Write dataframe to plain text scalar table format

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

def parse(text, shebang='#'):
  """Parse text of a scalar.dat file, should be table format.

  Args:
    text (str): content of scalar.dat file
    shebang (str, optional): marker for header line, default "#"
  Return:
    pd.DataFrame: table data
  Example:
    >>> with open('vmc.s001.scalar.dat', 'r') as f:
    >>>   text = f.read()
    >>>   df = parse(text)
  """
  fp = get_string_io(text)
  # try to read header line
  header = fp.readline().strip()
  fp.seek(0)
  # read data
  sep = r'\s+'
  if header.startswith(shebang):
    df = pd.read_csv(fp, sep=sep)
    # drop shebang from column names
    ncol_to_drop = len(shebang.split())
    print(shebang, ncol_to_drop)
    columns = df.columns
    df.drop(columns[-ncol_to_drop:], axis=1, inplace=True)
    df.columns = columns[ncol_to_drop:]
    # calculate local energy variance if possible (QMCPACK specific)
    if ('LocalEnergy' in columns) and ('LocalEnergy_sq' in columns):
      df['Variance'] = df['LocalEnergy_sq']-df['LocalEnergy']**2.
  else:
    df = pd.read_csv(fp, sep=sep, header=None)
  fp.close()
  # column labels should be strings
  df.columns = map(str, df.columns)
  return df

def read_to_list(dat_fname, **kwargs):
  """Read scalar.dat file into a list of pandas DataFrames

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
  lines = text.split('\n')
  if len(idxl) == 0:  # no header
    return [parse(text, **kwargs)]
  idxl.append(None)
  # now read data and use headers to label columns
  lines = text.split('\n')
  dfl = []
  for bidx, eidx in zip(idxl[:-1], idxl[1:]):
    text1 = '\n'.join(lines[bidx:eidx])
    df1 = parse(text1, **kwargs)
    dfl.append(df1)
  return dfl

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
    except AttributeError as err:
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

def error(trace, kappa=None):
  """Calculate the error of a trace of scalar data

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
    kappa (float,optional): auto-correlation time, default is to re-calculate
  Return:
    float: stderr, the error of the mean of this trace of scalars
  """
  from qharv.reel.forlib.stats import corr
  stddev = np.std(trace, ddof=1)
  if np.isclose(stddev, 0):  # easy case
    return 0.0  # no error for constant trace
  if kappa is None:  # no call to corr
    kappa = corr(trace)
  neffective = np.sqrt(len(trace)/kappa)
  err = stddev/neffective
  return err

def single_column(df, column, nequil):
  """Calculate mean and error of a column

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
  from qharv.reel.forlib.stats import corr
  myy = df[column].values[nequil:]

  ymean = np.mean(myy)
  ycorr = corr(myy)
  yerr  = error(myy, ycorr)
  return ymean, yerr, ycorr
