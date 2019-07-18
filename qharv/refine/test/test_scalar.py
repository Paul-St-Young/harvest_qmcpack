#!/usr/bin/env python
import numpy as np
from qharv.refine import scalar

def test_text_mean_error():
  ym  = np.array([-0.5048193, -0.50144785, -0.50485302, 14343])
  ye  = np.array([6.05058903e-05, 5.54368333e-05, 2.52821417e-05, 89])
  yt0 = np.array(['-0.50482(6)', '-0.50145(6)', '-0.50485(3)', '14343(89)'])

  yt = scalar.text_mean_error(ym, ye)
  for y, y0 in zip(yt, yt0):
    assert y.strip() == y0
