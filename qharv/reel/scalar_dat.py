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
