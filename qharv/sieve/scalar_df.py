# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to roughly process scalar Dataframes. Mostly built around pandas's API.
#
#  note: A scalar dataframe (scalar_df) is expected to contain the raw data, i.e.
# block-resolved expectation values, of a SINGLE calculation. If multiple runs are
# collected in the same dataframe, label by ['path','fdat'] and use groupby before
# applying the functions in this script.
import numpy as np
import pandas as pd
from qharv.reel.scalar_dat import error

def mean_error_scalar_df(df,nequil,kappa=None):
  """ get mean and average from a dataframe of raw scalar data (per-block) 
   take dataframe having columns ['LocalEnergy','Variance',...] to a 
   dataframe having columns ['LocalEnergy_mean','LocalEnergy_error',...]

   Args:
    df (pd.DataFrame): raw scalar dataframe, presumable generated using
     qharv.scalar_dat.parse with extra labels columns added to identify
     the different runs.
    nequil (int): number of equilibration blocks to throw out for each run.
   Returns:
    pd.DataFrame: mean_error dataframe 
  """
  sel = df['index'] > nequil

  # create pd.Series of mean
  msr = df.loc[sel].apply(np.mean).drop('index')

  # create pd.Series of error
  efunc = lambda x:error(x,kappa=kappa) # if kappa is not None, then 
  # auto-correlation is not re-calculated
  esr = df.loc[sel].apply( # error cannot be directly applied to matrix yet
    lambda x:float( np.apply_along_axis(efunc,0,x) )
  ).drop('index')

  df1 = msr.to_frame().T
  df2 = esr.to_frame().T
  jdf = df1.join(df2,lsuffix='_mean',rsuffix='_error')
  return jdf
# end def

def reblock(trace,block_size,min_nblock=4):
  """ block scalar trace to remove autocorrelation; see usage example in reblock_scalar_df
  Args:
    trace (np.array): a trace of scalars, may have multiple columns !!!! assuming leading dimension is the number of current blocks.
    block_size (int): size of block in units of current block.
    min_nblock (int,optional): minimum number of blocks needed for meaningful statistics, default is 4.
  Returns:
    np.array: re-blocked trace.
  """
  nblock= len(trace)//block_size
  nkeep = nblock*block_size
  if (nblock<min_nblock):
    raise RuntimeError('only %d blocks left after reblock'%nblock)
  # end if
  blocked_trace = trace[:nkeep].reshape(nblock,block_size,*trace.shape[1:])
  return np.mean(blocked_trace,axis=1)
# end def

def reblock_scalar_df(df,block_size,min_nblock=4):
  """ create a re-blocked scalar dataframe from a current scalar dataframe
   see reblock for details
  """
  return pd.DataFrame(
    reblock(df.values,block_size,min_nblock=min_nblock)
    ,columns=df.columns
  )
# end def
