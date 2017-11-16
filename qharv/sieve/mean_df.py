# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to further process mean dataframes. Mostly built around pandas's API.
#
# note: mean dataframes (mdf) are dataframes having the index of columns tructure as 
#  those returned by scalar_df.mean_error_scalar_df.
import numpy as np
import pandas as pd

def twist_average_mean_df(df0,drop_null=False):
  """ average scalar quantities over a set of calculations.
  The intented application is to average over a uniform grid of twist calculations.
  Args:
   df0 (pd.DataFrame): a mean dataframe containing ['path','fdat','*_mean','*_error'] columns.
    fdat is the scalar.dat filename. It will be used to extract 'series' and 'group' indices.
   drop_null (bool, optional): drop rows that contain any nan. These rows should be processed
    at a higher level (e.g. catach the RuntimeError).
  Returns:
    pd.DataFrame: df1, a mean dataframe containing one entry for each series.
     df1 is structually identical to a mean dataframe of a single twist.
  """
  # decide what to do with nans
  bad_sel = df0.isnull().any(axis=1)
  bad_entries = df0.loc[bad_sel]
  nbad = len(bad_entries)
  if (nbad>0) and (not drop_null):
    raise RuntimeError('%d bad runs found in scalar_df, set drop_null to drop bad data.'%nbad)
  # end if
  if (drop_null):
    df0.drop(bad_entries.index,inplace=True)
  # end if

  def sq_avg(trace): # not a well-written function ...
    """ average error """
    return np.sqrt( np.sum( trace**2. ) )/len(trace)
  # get metadata
  from qharv.reel import mole
  meta = df0['fdat'].apply(mole.interpret_qmcpack_fname).apply(pd.Series)
  df = pd.concat([meta,df0],axis=1)

  # average over twists
  cols = df.columns
  mcol = [col for col in cols if col.endswith('_mean')]
  ecol = [col for col in cols if col.endswith('_error')]
  rcol = [col for col in cols if (not col=='series') and\
            (not col.endswith('_mean')) and (not col.endswith('_error'))]
  groups = df.groupby('series')
  mdf = groups[mcol].apply(np.mean)
  edf = groups[ecol].apply(sq_avg)
  rdf = groups[rcol].apply(lambda x:x.drop(
   ['fdat','group'],axis=1).drop_duplicates().squeeze())

  df1 = rdf.join(mdf).join(edf).reset_index().sort_values('series')
  return df1
# end def twist_average_mean_df
