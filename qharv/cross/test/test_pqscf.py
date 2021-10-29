import numpy as np

def test_calc_spin_sph():
  from qharv.cross.pqscf import calc_spin_sph
  spin_wavefunctions = [
    (1, 0),  # spin up
    (0, 1),  # spin dn
    (1, 1),
    (1, 1j),
  ]
  expected_answers = [
    (1, 0, 0),  # (mag., theta, phi)
    (1, np.pi, 0),
    (2**0.5, np.pi/2, 0),
    (2**0.5, np.pi/2, np.pi/2),
  ]
  for s, e in zip(spin_wavefunctions, expected_answers):
    swf = np.array(s)
    ret = calc_spin_sph(swf)
    assert np.allclose(ret, e)

def test_rotate_spin():
  from qharv.cross.pqscf import calc_spin_sph
  from qharv.cross.pqscf import rotate_spin
  # start with spin up
  swf = np.array((1, 0))
  theta = np.pi/3
  phi = np.pi/4
  rmat = rotate_spin(theta, phi)
  s1 = np.dot(rmat, swf)
  m1, t1, p1 = calc_spin_sph(s1)
  assert np.isclose(t1, theta)
  assert np.isclose(p1, phi)
  # start with spin dn
  swf = np.array((0, 1))
  m0, t0, p0 = calc_spin_sph(swf)
  rmat0 = rotate_spin(-t0, -p0)
  s0 = np.dot(rmat0, swf)
  s1 = np.dot(rmat, s0)
  m1, t1, p1 = calc_spin_sph(s1)
  assert np.isclose(t1, theta)
  assert np.isclose(p1, phi)

if __name__ == '__main__':
  test_calc_spin_sph()
  test_rotate_spin()
