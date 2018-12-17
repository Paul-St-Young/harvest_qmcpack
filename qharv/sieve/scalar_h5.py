import os
import h5py
import numpy as np

def twist_average_h5(fh5, weights=None, **suffix_kwargs):
  """ twist average data in an HDF5 archive

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
  twist0 = fp.keys()[0]
  ymean, yerror = get_ymean_yerror(fp, twist0, **suffix_kwargs)
  # treat all other entries as metadata
  meta = {}
  for name in fp[twist0].keys():
    if name not in [ymean, yerror]:
      meta[name] = fp[os.path.join(twist0, name)].value
  # extract all ymean and yerror
  yml = []
  yel = []
  twists = fp.keys()
  # !!!! make sure twists are sorted
  itwists = [int(t.replace('twist', '')) for t in twists]
  assert sorted(itwists)
  ntwist = len(twists)
  if weights is None:
    weights = np.ones(ntwist)
  else:
    if len(weights) != ntwist:
      raise RuntimeError('%d weights for %d twists' % (len(weights), ntwist))
  wtot = weights.sum()
  for twist in twists:
    mpath = os.path.join(twist, ymean)
    epath = os.path.join(twist, yerror)
    yml.append(fp[mpath].value)
    yel.append(fp[epath].value)
  fp.close()
  yma = np.array(yml)
  yea = np.array(yel)
  # twist average with weights
  try:
    ym = np.sum(weights[:, np.newaxis, np.newaxis]*yma, axis=0)/wtot
    ye = np.sum(weights[:, np.newaxis, np.newaxis]*yea**2, axis=0)**0.5/wtot
  except:
    ym = np.sum(weights[:, np.newaxis]*yma, axis=0)/wtot
    ye = np.sum(weights[:, np.newaxis]*yea**2, axis=0)**0.5/wtot
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

def twist_concat_h5(fh5, name):
  fp = h5py.File(fh5)
  data = []
  for twist in fp:
    path = os.path.join(twist, name)
    data.append(fp[path].value)
  fp.close()
  return np.concatenate(data)
