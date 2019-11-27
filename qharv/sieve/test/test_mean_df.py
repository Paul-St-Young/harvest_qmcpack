import numpy as np

def read_test_data():
  import tarfile
  from qharv.reel.scalar_dat import parse
  ftar = 'li61a_bfd-n54-t8s1-li38s1.tar.gz'
  tar = tarfile.open(ftar, 'r:gz')
  mem = tar.next()  # only 1 member
  f = tar.extractfile(mem)
  df = parse(f.read())
  return df

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
