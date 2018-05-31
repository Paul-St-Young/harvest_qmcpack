import h5py
import numpy as np
from qharv.reel import config_h5


def test_saveh5():
  mat = np.arange(9).reshape(3, 3)
  config_h5.saveh5('mat.h5', mat)

  config_h5.saveh5('mat.h5', mat, name='mat')
  fp = h5py.File('mat.h5', 'r')
  mat1 = fp['mat'].value
  fp.close()
  assert np.allclose(mat, mat1)
