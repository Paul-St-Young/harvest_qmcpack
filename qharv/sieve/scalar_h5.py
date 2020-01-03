import os
import h5py
import numpy as np

def extract_twists(fh5, **suffix_kwargs):
  """Extract an observable at all twists from an HDF5 archive

  each twist should be a group in at root

  example:
    twist000
      myr
      gr_mean
      gr_error
    twist001
      myr
      gr_mean
      gr_error

  Args:
    fh5 (str): h5 file location
  Return:
    (dict, np.array, np.array): (meta data, mean, error)
  """
  fp = h5py.File(fh5)
  # determine ymean, yerror from first twist
  twist0 = list(fp.keys())[0]
  ymean, yerror = get_ymean_yerror(fp, twist0, **suffix_kwargs)
  # treat all other entries as metadata
  meta = {}
  for name in fp[twist0].keys():
    if name not in [ymean, yerror]:
      meta[name] = fp[os.path.join(twist0, name)][()]
  # extract all ymean and yerror
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
  fp.close()
  yma = np.array(yml)  # mean
  yea = np.array(yel)  # error
  return meta, yma, yea

def twist_average_h5(fh5, weights=None, **suffix_kwargs):
  """ twist average data in an HDF5 archive

  see extract_twists for h5 file format

  Args:
    fh5 (str): h5 file location
  Return:
    (dict, np.array, np.array): (meta data, mean, error)
  """
  from qharv.sieve.mean_df import taw
  meta, yma, yea = extract_twists(fh5, **suffix_kwargs)
  ntwist = len(yma)
  # twist average with weights
  if weights is None:
    weights = np.ones(ntwist)
  else:
    if len(weights) != ntwist:
      raise RuntimeError('%d weights for %d twists' % (len(weights), ntwist))
  ym, ye = taw(yma, yea, weights)
  return meta, ym, ye

def get_ymean_yerror(fp, twist0, msuffix='_mean', esuffix='_error'):
  ymean = None
  yerror = None
  for name in fp[twist0].keys():
    if name.endswith(msuffix):
      ymean = name
    if name.endswith(esuffix):
      yerror = name
  if ymean is None:
    raise RuntimeError('no entry with suffix %s' % msuffix)
  if yerror is None:
    raise RuntimeError('no entry with suffix %s' % esuffix)
  ynamem = '_'.join(ymean.split('_')[:-1])
  ynamee = '_'.join(yerror.split('_')[:-1])
  if ynamem != ynamee:
    raise RuntimeError('yname mismatch')
  return ymean, yerror

def twist_concat_h5(fh5, name, twists=None):
  fp = h5py.File(fh5)
  if twists is None:
    twists = fp.keys()
  data = []
  for twist in twists:
    path = os.path.join(twist, name)
    val = fp[path][()]
    data.append(val)
  fp.close()
  return np.concatenate(data, axis=0)
