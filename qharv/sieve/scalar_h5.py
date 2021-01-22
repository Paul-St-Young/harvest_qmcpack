import os
import h5py
import numpy as np

def extract_twists(fh5, **suffix_kwargs):
  """Extract an observable at all twists from an HDF5 archive

  each twist should be a group in at root

  example fh5:
    twist000
      myr
      gr_mean
      gr_error
      sk_mean
      sk_error
    twist001
      myr
      gr_mean
      gr_error
      sk_mean
      sk_error
  return:
    {'myr': [], 'gr_mean': [], 'gr_error': [], 'sk_mean': [], 'sk_error': []}

  Args:
    fh5 (str): h5 file location
  Return:
    dict: (meta, mean, error)
  """
  from qharv.sieve.mean_df import categorize_columns
  fp = h5py.File(fh5, 'r')
  # determine meta, ymean, yerror from first twist
  twist0 = list(fp.keys())[0]
  cols = fp[twist0].keys()
  labels, mcols, ecols = categorize_columns(cols)
  # treat all other entries as metadata
  meta = {}
  for name in labels:
    meta[name] = fp[os.path.join(twist0, name)][()]
  # extract all ymean and yerror
  data = {}
  for ymean, yerror in zip(mcols, ecols):
    yml = []
    yel = []
    twists = fp.keys()
    # !!!! make sure twists are sorted
    itwists = [int(t.replace('twist', '')) for t in twists]
    assert sorted(itwists)
    ntwist = len(twists)
    for twist in twists:
      mpath = os.path.join(twist, ymean)
      ym1 = fp[mpath][()]
      epath = os.path.join(twist, yerror)
      ye1 = fp[epath][()]
      yml.append(ym1)
      yel.append(ye1)
    yma = np.array(yml)  # mean
    yea = np.array(yel)  # error
    data[ymean] = yma
    data[yerror] = yea
  fp.close()
  data.update(meta)
  return data

def twist_average_h5(fh5, weights=None, **suffix_kwargs):
  """ twist average data in an HDF5 archive

  see extract_twists for h5 file format

  Args:
    fh5 (str): h5 file location
  Return:
    (dict, np.array, np.array): (meta data, mean, error)
  """
  from qharv.sieve.mean_df import taw, categorize_columns
  data = extract_twists(fh5, **suffix_kwargs)
  labels, mcols, ecols = categorize_columns(data.keys())
  ntwist = len(data[mcols[0]])
  # twist average with weights
  if weights is None:
    weights = np.ones(ntwist)
  else:
    if len(weights) != ntwist:
      raise RuntimeError('%d weights for %d twists' % (len(weights), ntwist))
  # twist average mean, error columns
  for ymean, yerror in zip(mcols, ecols):
    yma = data[ymean]
    yea = data[yerror]
    ym, ye = taw(yma, yea, weights)
    data[ymean] = ym
    data[yerror] = ye
  return data

def twist_concat_h5(fh5, name, twists=None):
  fp = h5py.File(fh5, 'r')
  if twists is None:
    twists = fp.keys()
  data = []
  for twist in twists:
    path = os.path.join(twist, name)
    val = fp[path][()]
    data.append(val)
  fp.close()
  return np.concatenate(data, axis=0)
