import numpy as np

def example_pol_cart():
  pol = np.array([
    [0.716242, 88.624935, 2.410918],
    [0.716130, 92.127722, 176.742020],
  ])
  cart = np.array([
    [ 0.71540193,  0.03012079, 0.01718774],
    [-0.71447963,  0.04067093, -0.02658792],
  ])
  return pol, cart

def test_polar2cart():
  from qharv.cross.pwscf import polar2cart
  pol, cart0 = example_pol_cart()
  cart = polar2cart(pol)
  assert np.allclose(cart, cart0)

def test_cart2polar():
  from qharv.cross.pwscf import cart2polar
  pol0, cart = example_pol_cart()
  pol = cart2polar(cart)
  assert np.allclose(pol, pol0)
