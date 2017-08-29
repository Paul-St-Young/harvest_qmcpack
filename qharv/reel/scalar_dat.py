import numpy as np
import pandas as pd

def parse(dat_fname):
  """ read the scalar.dat file, should be table format readable by numpy.loadtxt.
   The header line should start with '#' and contain column labels.
  Args:
    dat_fname (str): name of input file
  Returns:
    df (pd.DataFrame): table of data, effect: self.df=df """

  # pandas's equivalent of numpy.loadtxt
  df = pd.read_csv(dat_fname,sep='\s+')

  # remove first column name '#'
  columns = df.columns
  df.drop(columns[-1],axis=1,inplace=True)
  df.columns = columns[1:]

  return df
# end def parse

def corr(trace):
  """ calculate the autocorrelation of a trace of scalar data
  Args:
    trace (list): should be a 1D iterable array of floating point numbers
  Returns: r
    correlation_time (float): return the autocorrelation time of this trace of scalars
  """
 
  mu     = np.mean(trace)
  stddev = np.std(trace,ddof=1)
 
  correlation_time = 0.
  for k in range(1,len(trace)):
      # calculate auto_correlation
      auto_correlation = 0.0
      num = len(trace)-k
      for i in range(num):
          auto_correlation += (trace[i]-mu)*(trace[i+k]-mu)
      # end for i
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

def single_column(df,column,nequil):
  """ calculate mean and error of a column
  Args:
    df (pd.DataFrame): table of data (e.g. from scalar_dat.parse)
    column (str): name of column
    nequil (int): number of equilibration blocks
  Returns:
    ymean (float): mean of column
    yerr  (float): error of column
  """

  myx = df['index'].values
  myy = df[column].values[nequil:]

  ymean = np.mean(myy)
  yerr  = np.std(myy)/np.sqrt(len(myy)/corr(myy))
  return ymean,yerr
# end def

