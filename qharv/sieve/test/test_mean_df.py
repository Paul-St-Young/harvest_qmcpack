import numpy as np
import pandas as pd

def read_test_data():
  import os
  path = os.path.dirname(os.path.relpath(__file__))
  import tarfile
  from qharv.reel.scalar_dat import parse
  ftar = os.path.join(path, 'li61a_bfd-n54-t8s1-li38s1.tar.gz')
  tar = tarfile.open(ftar, 'r:gz')
  mem = tar.next()  # only 1 member
  f = tar.extractfile(mem)
  df = parse(f.read())
  return df

def get_test_results():
  results = {  # label by series 0 (VMC), 1, 2, 3 (DMC)
    0: {
      'LocalEnergy_mean': -13.845735734374998,
      'LocalEnergy_error': 0.0015776,
      'Variance_mean': 0.205234,
      'Variance_error': 0.002233,
    },
    1: {
      'LocalEnergy_mean': -13.930791,
      'LocalEnergy_error': 0.000937,
      'Variance_mean': 0.238057,
      'Variance_error': 0.001881,
    },
    2: {
      'LocalEnergy_mean': -13.916134,
      'LocalEnergy_error': 0.0007,
      'Variance_mean': 0.224522,
      'Variance_error': 0.000912,
    },
    3: {
      'LocalEnergy_mean': -13.910996,
      'LocalEnergy_error': 0.000582,
      'Variance_mean': 0.218816,
      'Variance_error': 0.000654
    },
  }
  return results

def test_categorize_columns():
  from qharv.sieve.mean_df import categorize_columns
  df = read_test_data()
  ret = categorize_columns(df)
  # check exact columns
  names = ret[0]
  ref_names = ['series', 'group', 'nelec', 'weights', 'timestep', 'acc']
  for name in ref_names:
    assert name in names
  # check mean and error columns
  mcols = ret[1]
  ecols = ret[2]
  msuf = '_mean'
  esuf = '_error'
  ref_cols = ['LocalEnergy', 'Variance', 'Kinetic']
  for mcol, ecol in zip(mcols, ecols):
    assert msuf in mcol
    assert esuf in ecol
    col = mcol.replace(msuf, '')
    col1 = ecol.replace(esuf, '')
    assert col == col1
    assert col in ref_cols

def test_twist_average_with_weights():
  from qharv.sieve.mean_df import taw
  df = read_test_data()
  sel = df.series == 0
  mydf = df.loc[sel]
  wts = mydf.weights.values

  # no errorbar
  ym = mydf.nelec.values
  ye = np.zeros(len(wts))
  y0 = taw(ym, ye, wts)[0]

  # check by making copies
  ntwists = (wts*32).astype(int)
  ntot = ntwists.sum()
  assert ntot == 64
  tot = 0.0
  for itwist, ncopy in enumerate(ntwists):
    for icopy in range(ncopy):
      yval = ym[itwist]
      tot += yval
  expect = tot/ntot
  assert np.isclose(y0, expect)

  # with errorbar
  ym = mydf.LocalEnergy_mean.values
  ye = mydf.LocalEnergy_error.values
  y0m, y0e = taw(ym, ye, wts)
  # check by making copies
  tot = 0.0
  terr = 0.0
  nterm = 0
  for itwist, ncopy in enumerate(ntwists):
    for icopy in range(ncopy):
      yval = ym[itwist]
      tot += yval
      yerr = ye[itwist]
      # spread statistics out over ncopy twists
      terr += yerr**2*ncopy
      nterm += 1
  assert nterm == ntot
  em = tot/ntot
  ee = terr**0.5/ntot
  assert np.isclose(y0m, em)
  assert np.isclose(y0e, ee)

  # regression test
  results = get_test_results()
  em0 = results[0]['LocalEnergy_mean']
  ee0 = results[0]['LocalEnergy_error']
  assert np.isclose(em0, em)
  assert np.isclose(ee0, ee)

def test_twist_average_mean_df():
  from qharv.sieve.mean_df import dfme
  df = read_test_data()
  # check no error
  ndf = df.groupby('series').apply(dfme,
    ['nelec'], no_error=True, weight_name='weights')
  assert np.allclose(ndf.nelec.values, 53.90625)
  # check with error
  atol = 1e-6
  results = get_test_results()
  rdf = pd.DataFrame(results).T
  mdf = df.groupby('series').apply(dfme,
    ['LocalEnergy', 'Variance'], weight_name='weights')
  for series in mdf.index:
    entry = mdf.loc[series]
    entry0 = rdf.loc[series]
    for key, val0 in entry0.items():
      val = entry[key]
      assert np.isclose(val, val0, atol=atol)
