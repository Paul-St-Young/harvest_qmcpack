# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse scalar table output. Mostly built around pandas's API.

import numpy as np
import pandas as pd

def parse(dat_fname):
  """ read the scalar.dat file, should be table format readable by numpy.loadtxt.

   The header line should start with '#' and contain column labels.

  Args:
    dat_fname (str): name of input file
  Returns:
    pd.DataFrame: df containing the table of data """

  # pandas's equivalent of numpy.loadtxt
  df = pd.read_csv(dat_fname,sep='\s+')

  # remove first column name '#'
  columns = df.columns
  df.drop(columns[-1],axis=1,inplace=True)
  df.columns = columns[1:]

  # calculate local energy variance if possible
  if ('LocalEnergy' in columns) and ('LocalEnergy_sq' in columns):
    df['Variance'] = df['LocalEnergy_sq']-df['LocalEnergy']**2.
  # end if

  return df
# end def parse

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

  nequil blocks of data are thrown out; autocorrelation time is taken into account when calculating error
  The equilibrated data is assumed to have Gaussian distribution. Error is calculated for one standard deviation (1-sigma error).

  Args:
    df (pd.DataFrame): table of data (e.g. from scalar_dat.parse)
    column (str): name of column
    nequil (int): number of equilibration blocks
  Returns:
    (float,float): (ymean,yerr), where ymean is the mean of column, while yerr is the 1-sigma error of column
  """

  myx = df['index'].values
  myy = df[column].values[nequil:]

  ymean = np.mean(myy)
  yerr  = np.std(myy)/np.sqrt(len(myy)/corr(myy))
  return ymean,yerr
# end def
