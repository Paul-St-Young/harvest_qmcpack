# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to gather dataframes.
import numpy as np
import pandas as pd

def check_complete(fxml, group=None):
  """Check if a run has finished.

  Input:
    fxml (str): input file
    group (int, optional): group index
  Return:
    bool: True if run is complete
  """
  import os
  from qharv.seed import xml, qmcpack_in
  path = os.path.dirname(fxml)
  doc = xml.read(fxml)
  pm = qmcpack_in.output_prefix_meta(doc, group=group)
  qmcs = doc.findall('.//qmc')
  series_complete = np.zeros(len(qmcs), dtype=bool)
  for iq, (prefix, meta) in enumerate(pm.items()):
    # expected number of blocks
    qmc = qmcs[iq]
    nblock0 = int(xml.get_param(qmc, 'blocks'))
    # check number of printed blocks
    fname = '%s.scalar.dat' % prefix
    fsca = os.path.join(path, fname)
    if not os.path.isfile(fsca):
      break
    nblock = sum(1 for line in open(fsca, 'r'))-1
    # decide if this series was complete
    series_complete[iq] = nblock == nblock0
  is_complete = np.all(series_complete)
  return is_complete

def scalar_dat(fxml, nequil, group=None, suffix='scalar.dat'):
  """Gather scalar.dat files produced by input fxml.

  Inputs:
    fxml (str): input file
    nequil (int or list): if list, then must have one int per series
    group (int, optional): if provided, then add '.g%03d' % group to prefix
    suffix (str, optional): default 'scalar.dat'
  Output:
    pd.DataFrame: statistics of each run
  Examples:
    $ ls
      vmc.xml qmc.s000.scalar.dat
    >>> scalar_dat('vmc.xml', 8)  # throw out 8 blocks
    $ ls
      dmc.xml qmc.s000.scalar.dat qmc.s001.scalar.dat
    >>> scalar_dat('dmc.xml', [8, 16])  # throw out 8 in VMC, 16 in DMC
    $ ls
      twist1.xml qmc.g001.s000.scalar.dat qmc.g001.s001.scalar.dat
    >>> scalar_dat('twist1.xml', [8, 16], group=1)
  """
  import os
  from qharv.seed import xml, qmcpack_in
  from qharv.reel import scalar_dat, mole
  from qharv.sieve import mean_df
  path = os.path.dirname(fxml)
  doc = xml.read(fxml)
  pm = qmcpack_in.output_prefix_meta(doc, group=group)
  # decide equilibration length from input
  if np.issubdtype(type(nequil), np.integer):
    neql = [nequil]*len(pm)
  else:  # one equilibration length for each series
    neql = nequil
    if len(neql) != len(pm):
      msg = '%d nequil provided for %d series' % (len(neql), len(pm))
      raise RuntimeError(msg)
  # add more metadata
  myid, s0 = xml.get_id_series(doc)
  nelecs = xml.get_nelecs(doc.getroot())
  nelec = sum(nelecs.values())
  for (prefix, meta), nequil in zip(pm.items(), neql):
    meta['id'] = myid
    meta['nequil'] = nequil
    meta['nelec'] = nelec
  # gather each series
  dfl = []
  for prefix, meta in pm.items():
    # read data table file
    fsca_dat = '%s.%s' % (prefix, suffix)
    floc = os.path.join(path, fsca_dat)
    df1 = scalar_dat.read(floc)
    # calculate equilibration length
    nequil = meta['nequil']
    # discard equilibration and average over projection trajectory
    mdf = mean_df.create(df1.iloc[nequil:])
    # add metadata
    mdf['path'] = mole.clean_path(path)
    mdf['nblock'] = len(df1)
    for key, val in meta.items():
      mdf[key] = val
    dfl.append(mdf)
  df = pd.concat(dfl, axis=0).reset_index(drop=True)
  convert_known_metadata_types(df)
  return df

def convert_known_metadata_types(df):
  # known metadata type
  col_types = dict(
    timestep = float,
    target_walkers = int,
    samples = int,
  )
  for col in df.columns:
    sel = ~df[col].isnull()
    if col in col_types:
      dtype = col_types[col]
      df.loc[sel, col] = df.loc[sel, col].astype(dtype)
