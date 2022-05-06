import numpy as np
from qharv.sieve import mean_df

def data_with_noise(norder=1):
  iseed = 42
  sig = 1e-3
  nx = 3
  x = np.array(range(1, nx+1))
  ym = np.array(range(1, nx+1), dtype=float)
  ym = ym**norder
  np.random.seed(iseed)
  noise = sig*np.random.randn(nx)
  ym += noise
  ye = sig**np.ones(nx)
  return x, ym, ye

def test_linear_no_error():
  x = np.array([1, 2, 3])
  y = np.array([1, 2, 3])
  y0 = mean_df.polyextrap(x, y)
  assert np.isclose(y0, 0)
  y1 = mean_df.polyextrap(x, y, xtarget=1)
  assert np.isclose(y1, 1)

def test_linear_with_error():
  nsig = 3
  x, ym, ye = data_with_noise()
  y0m, y0e = mean_df.polyextrap(x, ym, ye)
  assert (y0m-nsig*y0e < 0) & (0 < y0m+nsig*y0e)
  assert np.isclose(y0m, 0.0001844)
  assert np.isclose(y0e, 0.00152759)

def test_quadratic_with_error():
  nsig = 3
  x, ym, ye = data_with_noise(norder=2)
  y0m, y0e = mean_df.polyextrap(x, ym, ye, 2)
  assert (y0m-nsig*y0e < 0) & (0 < y0m+nsig*y0e)
  assert np.isclose(y0m, 0.0025526239)
  assert np.isclose(y0e, 0.00435896876)

if __name__ == '__main__':
  test_quadratic_with_error()
