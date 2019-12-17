import numpy as np

def test_ft_iso3d():
  from qharv.inspect.grsk import ft_iso3d, ift_iso3d
  finex = np.linspace(1e-6, 10, 256)
  sig = 1.

  # Gaussian in real space
  finey = np.exp(-finex**2/(2*sig**2))/(np.sqrt(2*np.pi)*sig)
  assert np.isclose(np.trapz(finey, finex), .5, 1e-5)

  # FT to Gaussian in reciprocal space
  finek = finex
  yk = ft_iso3d(finek, finex, finey)

  # iFT Gaussian back
  y1 = ift_iso3d(finex, finek, yk)
  assert np.isclose(np.trapz(y1, finex), .5, 1e-5)
  assert np.allclose(y1, finey)
