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

def test_cell_parameters():
  from qharv.cross.pwscf import cell_parameters
  axes = np.eye(3)
  axes[1, 0] = -0.5
  text = cell_parameters(axes, unit='angstrom', fmt='%.2f')
  text0 = '''\nCELL_PARAMETERS angstrom
1.00 0.00 0.00 
-0.50 1.00 0.00 
0.00 0.00 1.00 
'''
  assert text == text0

def test_parse_cell():
  from qharv.cross.pwscf import cell_parameters, parse_cell_parameters
  axes = np.eye(3)
  axes[1, 0] = -0.5
  unit = 'angstrom'
  text = cell_parameters(axes, unit=unit, fmt='%.2f')
  unit1, axes1 = parse_cell_parameters(text)
  assert np.allclose(axes1, axes)
  assert unit1 == unit
