# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse scalar table output. Mostly built around pandas's API.

import numpy as np
import pandas as pd

def parse(dat_fname):
  """ for backwards compatibility, please use read() instead!

  for future reference: read() should read from a file on disk
    parse() should parse text or another internal rep. in memory
  """
  try:
    return read(dat_fname)
  except IOError:
    raise NotImplementedError('text parsing not yet implemented')

def read(dat_fname):
  """ read the scalar.dat file, should be table format readable by numpy.loadtxt.

   The header line should start with '#' and contain column labels.

  Args:
    dat_fname (str): name of input file
  Returns:
    pd.DataFrame: df containing the table of data
  """

  # check if a header line exists, if it does not, then set header=None
  with open(dat_fname,'r') as fp:
    header = fp.readline().strip()
  # end with
  
  if not header.startswith('#'): # there is no header
    # pandas's equivalent of numpy.loadtxt
    df = pd.read_csv(dat_fname,sep='\s+',header=None)
  else: # do some extra parsing of the header
    df = pd.read_csv(dat_fname,sep='\s+')

    # remove first column name '#'
    columns = df.columns
    df.drop(columns[-1],axis=1,inplace=True)
    df.columns = columns[1:]

    # calculate local energy variance if possible
    if ('LocalEnergy' in columns) and ('LocalEnergy_sq' in columns):
      df['Variance'] = df['LocalEnergy_sq']-df['LocalEnergy']**2.
    # end if
  # end if

  # column labels should be strings
  df.columns = map(str,df.columns)

  return df
# end def parse

def read_to_list(dat_fname):
  """ read scalar.dat file into a list of pandas DataFrames

  Header lines should start with '#', assumed to contain column labels.
  Many scalar.dat files can be concatenated. A list will be returned.

  Args:
    dat_fname (str): name of input file
  Return:
    list: list of df(s) containing the table(s) of data
  """
  # first separate out the header lines and parse them
  from StringIO import StringIO
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
    df1 = pd.read_csv(StringIO(text1), sep='\s+', header=None)
    df1.columns = columns
    dfl.append(df1)
  return dfl

def write(dat_fname, df, header_pad='# ', **kwargs):
  """ write dataframe to plain text scalar table format

  Lightly wrap around pandas.to_string with defaults to index and float_format

  Args:
    dat_fname (str): output data file name
    df (pd.DataFrame): data
    header_pad (str, optional): pad beginning of header with comment string, default '# '
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

  correlation time is defined as the integral of the auto-correlation function from t=0 to when the function first reaches 0.

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
  Returns:
    float: correlation_time, the autocorrelation time of this trace of scalars
  """
 
  mu     = np.mean(trace)
  stddev = np.std(trace,ddof=1)
  if np.isclose(stddev,0): # easy case
    return np.inf # infinite correlation for constant trace
  # end if

  correlation_time = 0.
  for k in range(1,len(trace)):
    # calculate auto_correlation
    auto_correlation = 0.0
    num = len(trace)-k
    auto_correlation = np.dot(trace[:num]-mu, trace[k:]-mu)
    auto_correlation *= 1.0/(num*stddev**2)
    if auto_correlation > 0:
      correlation_time += auto_correlation
    else:
      break
    # end if
  # end for k
 
  correlation_time = 1.0 + 2.0*correlation_time
  return correlation_time
# end def corr
def error(trace,kappa=None):
  """ calculate the error of a trace of scalar data

  Args:
    trace (list): should be a 1D iterable array of floating point numbers
    kappa (float,optional): auto-correlation time, default is to re-calculate 
  Returns:
    float: stderr, the error of the mean of this trace of scalars
  """
  stddev = np.std(trace,ddof=1)
  if np.isclose(stddev,0): # easy case
    return 0.0 # no error for constant trace
  # end if

  if kappa is None: # no call to corr 
    kappa = corr(trace)
  # end if
  neffective = np.sqrt(len(trace)/kappa)
  err = stddev/neffective
  return err
# end def error

def single_column(df,column,nequil):
  """ calculate mean and error of a column

    nequil blocks of data are thrown out; autocorrelation time is taken into 
  account when calculating error. The equilibrated data is assumed to have 
  Gaussian distribution. Error is calculated for one standard deviation
  (1-sigma error).

  Args:
    df (pd.DataFrame): table of data (e.g. from scalar_dat.parse)
    column (str): name of column
    nequil (int): number of equilibration blocks
  Returns:
    (float,float,float): (ymean,yerr,ycorr), where ymean is the mean of column
     , yerr is the 1-sigma error of column, and ycorr is the autocorrelation
  """

  myy = df[column].values[nequil:]

  ymean = np.mean(myy)
  ycorr = corr(myy)
  yerr  = error(myy,ycorr)

  return ymean,yerr,ycorr
# end def
