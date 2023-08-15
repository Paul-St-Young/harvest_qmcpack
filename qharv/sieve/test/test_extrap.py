import numpy as np
from qharv.sieve.extrap import polyextrap, polyfit

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
  y0 = polyextrap(x, y)
  assert np.isclose(y0, 0)
  y1 = polyextrap(x, y, xtarget=1)
  assert np.isclose(y1, 1)

def test_linear_with_error():
  nsig = 3
  x, ym, ye = data_with_noise()
  y0m, y0e = polyextrap(x, ym, ye)
  print(y0m)
  print(y0e)
  assert (y0m-nsig*y0e < 0) & (0 < y0m+nsig*y0e)
  assert np.isclose(y0m, 0.0001844)
  assert np.isclose(y0e, 0.0015275)

def test_quadratic_with_error():
  nsig = 3
  x, ym, ye = data_with_noise(norder=2)
  y0m, y0e = polyextrap(x, ym, ye, 2)
  assert (y0m-nsig*y0e < 0) & (0 < y0m+nsig*y0e)
  assert np.isclose(y0m, 0.0025526239)
  assert np.isclose(y0e, 0.00435889879)

def test_linear_polyfit_flat_with_error():
  x = [0.5, 0.25]
  ym = [-3.1039024668401796, -3.1039024931067947]
  ye = [0.0001668663143609, 0.0002086231786522]
  popt, perr = polyfit(x, ym, ye, 1)
  assert np.isclose(popt[1], -3.10390252)
  assert np.isclose(perr[1], 0.00044938)

if __name__ == '__main__':
  test_linear_polyfit_flat_with_error()
