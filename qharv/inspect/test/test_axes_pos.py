#!/usr/bin/env python
import numpy as np
from qharv.inspect import axes_pos

axes0 = np.array([
  [2.46637878, -4.27189335, 0.],
  [2.46637878,  4.27189335, 0.],
  [0.,          0.,         8.45173341]
])

pos0 = np.array([
  [1.56156541,  1.57330776, 7.42305655],
  [1.56156506,  2.69858287, 3.19719163],
  [3.37119214, -2.69858455, 5.25453794]
])

dists0 = [4.37311998, 2.26578821, 2.43520807]

# basic 3D lattice vectors
sc0 = np.eye(3)
bcc0 = 0.5*(np.ones(3)-2*np.eye(3))
fcc0 = 0.5*(np.ones(3)-np.eye(3))
# basic 2D lattice vectors
tri0 = 0.5*np.array([
  [3**0.5,  1],
  [3**0.5, -1]
])

def test_abc():
  axes_to_test = [
    axes0, bcc0, fcc0
  ]
  abc_refs = [
    (4.932757553357808, 4.932757553357808, 8.45173341),
    (0.8660254037844386, 0.8660254037844386, 0.8660254037844386),
    (0.7071067811865476, 0.7071067811865476, 0.7071067811865476)
  ]
  for axes, abc0 in zip(axes_to_test, abc_refs):
    abc = axes_pos.abc(axes)
    assert np.allclose(abc, abc0)

def test_displacement():
  from itertools import combinations
  nptcl = len(pos0)
  disp_table = [axes_pos.displacement(axes0, pos0[i], pos0[j])
    for (i, j) in combinations(range(nptcl), 2)]
  dists = np.linalg.norm(disp_table, axis=1)
  assert np.allclose(dists, dists0)

def test_auto_distance_table():
  dtable = axes_pos.auto_distance_table(axes0, pos0)

  nptcl  = len(pos0)
  i_triu = np.triu_indices(nptcl, m=nptcl, k=1)
  dists  = dtable[i_triu]
  assert np.allclose(dists, dists0)

def test_volume():
  # primitive cell of b.c.c. with lattice constant 1.0
  assert np.allclose(bcc0, 0.5*(np.ones(3)-2.*np.eye(3)))
  # primitive cell of f.c.c. with lattice constant 1.0
  assert np.allclose(fcc0, 0.5*(np.ones(3)-1.*np.eye(3)))
  # 2 atoms in b.c.c. conventional cell
  assert np.isclose(axes_pos.volume(bcc0), 0.5)
  # 4 atoms in f.c.c. conventional cell
  assert np.isclose(axes_pos.volume(fcc0), 0.25)
  # no negative volume
  axes_neg = np.array([
    [2.55252043e+00, 9.10248160e+00, -4.45451981e+00],
    [5.15835511e+00, 1.16509760e-03, -8.90903961e+00],
    [5.15835511e+00, 1.16509760e-03,  8.90903961e+00]
  ])
  assert np.isclose(axes_pos.volume(axes_neg), 836.574116485)

def test_raxes():
  b2f = axes_pos.raxes(bcc0)/(2.*np.pi)/2.
  assert np.allclose(b2f, fcc0)
  f2b = axes_pos.raxes(fcc0)/(2.*np.pi)/2.
  assert np.allclose(f2b, bcc0)

def test_rwsc():
  # !!!! THIS TEST IS WRONG !!!!
  # a fairly extreme slab-like cell
  axes = np.array([
   [7.73625309, 18.39193864, -13.47014157],
   [-0.08306451, 27.58607338,  17.96016072],
   [-2.59722820, -0.00040676,  13.47012604]
  ])
  # dn = 2 is NOT sufficient
  rwsc = axes_pos.rwsc(axes, dn=2)
  assert np.isclose(rwsc, 5.281986)

def axes0_pos0(rs):
  alat = np.sqrt(2*2*np.pi/3**0.5)*rs
  axes = alat*np.eye(3)
  axes[1, :] = [alat/2., 3**0.5*alat/2., 0]

  upos = np.array([
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.5]
  ])
  pos = np.dot(upos, axes)
  return axes, pos

def test_pos_in_box():
  axes, pos = axes0_pos0(50.)

  tmat = np.array([
    [1, 0, 0],
    [0, 3, 0],
    [0, 0, 1]
  ])
  axes1 = np.dot(tmat, axes)
  upos = np.dot(pos, np.linalg.inv(axes))
  cpos = axes_pos.cubic_pos(2)
  upos1 = np.concatenate([up+cpos for up in upos], axis=0)
  pos1 = np.dot(upos1, axes)

  pos2 = axes_pos.pos_in_axes(axes1, pos1, ztol=1e-8)
  upos2 = np.dot(pos2, np.linalg.inv(axes1))
  assert np.allclose(upos2[4:6], np.array([
    [0, 0, 0.5],
    [0, 0, 0.5]
  ]))

def test_cubic_pos():
  cpos1 = axes_pos.cubic_pos(2)
  cpos2 = axes_pos.cubic_pos([2, 2, 2])
  assert np.allclose(cpos1, cpos2)

def test_rins():
  for axes, r0 in zip(
    [sc0, bcc0, fcc0],
    [0.5, 0.3535534, 0.288675]
  ):
    assert np.isclose(axes_pos.rins(axes), r0, atol=1e-6)

def test_rins2d():
  for axes, r0 in zip(
    [tri0],
    [3**0.5/4]
  ):
    assert np.isclose(axes_pos.rins(axes), r0, atol=1e-6)

def test_rwsc2d():
  for axes, r0 in zip(
    [tri0],
    [0.5]
  ):
    assert np.isclose(axes_pos.rwsc(axes), r0, atol=1e-6)

def test_get_nvecs():
  mx = 3
  axes = axes0
  nvecs0 = axes_pos.cubic_pos(mx)
  pos = np.dot(nvecs0, axes)
  nvecs = axes_pos.get_nvecs(axes, pos)
  assert np.allclose(nvecs, nvecs0)

def test_find_dimers():
  axes = np.array([
    [5.7, 0.0, 0.0],
    [0.0, 6.6, 0.0],
    [0.0, 0.0, 6.6],
  ])
  pos = np.array([
    [1.3, 2.5, 0.2],
    [1.6, 0.4, 5.7],
    [1.1, 0.9, 5.8],
    [5.8, 4.5, 0.6],
    [4.0, 4.7, 5.5],
    [3.9, 4.5, 6.2],
    [3.2, 3.3, 2.3],
    [5.5, 1.6, 2.9],
    [1.3, 5.4, 1.1],
    [0.9, 5.7, 1.6],
    [0.1, 5.6, 6.5],
    [-0.0, 6.2, 6.3],
    [0.3, 1.9, 0.8],
    [5.8, 4.2, 0.0],
    [1.7, 3.3, 5.2],
    [1.1, 6.4, 3.0],
    [2.5, 5.7, 1.9],
    [2.9, 6.3, 1.8],
    [1.2, 5.9, 5.1],
    [0.9, 3.0, 5.4],
    [1.7, 1.4, 4.0],
    [1.6, 4.6, 0.1],
    [4.2, 1.3, 1.1],
    [3.9, 0.7, 1.6],
    [0.4, 1.2, 1.1],
    [1.6, 6.0, 3.4],
    [3.0, 0.3, 3.8],
    [2.4, 6.6, 4.0],
    [0.1, 0.2, 2.4],
    [1.9, 4.3, 2.4],
    [5.0, 3.4, 3.4],
    [-0.2, 3.3, 3.8],
    [3.6, 4.5, 1.1],
    [4.1, 3.8, 1.3],
    [4.4, 6.2, 0.1],
    [4.7, 0.8, 5.7],
    [5.1, 1.8, 2.3],
    [2.4, 1.8, 3.7],
    [4.5, 6.3, 4.8],
    [3.8, 6.0, 4.9],
    [3.2, 2.8, 3.0],
    [4.2, 6.4, 3.3],
    [0.3, 0.6, 4.9],
    [0.0, 0.3, 4.1],
    [5.1, 1.1, 6.3],
    [1.0, 3.1, 0.5],
    [0.3, 1.8, 4.3],
    [0.4, 2.4, 4.7],
    [4.5, 1.5, 4.8],
    [4.6, 1.7, 4.0],
    [-0.1, 3.7, 2.0],
    [0.7, 3.5, 2.0],
    [1.5, 1.8, 2.3],
    [1.4, 1.0, 2.3],
    [3.4, 1.2, 3.0],
    [3.2, 1.5, 2.4],
    [4.0, 5.2, 2.7],
    [4.2, 4.6, 3.1],
    [5.8, 6.6, 1.7],
    [1.7, 4.5, 5.9],
    [1.4, 2.7, 3.4],
    [1.6, 2.7, 2.7],
    [2.7, 4.0, 6.0],
    [3.0, 4.0, 0.1],
    [3.0, 2.8, 4.4],
    [3.7, 2.7, 4.7],
    [2.3, 5.6, -0.1],
    [2.1, 2.0, 5.5],
    [5.4, 5.2, 3.2],
    [3.8, -0.1, 0.2],
    [4.3, 3.2, 1.0],
    [4.7, 2.6, 0.6],
    [3.6, 1.9, 6.4],
    [4.0, 2.6, -0.2],
    [5.7, 4.6, 4.8],
    [5.1, 4.6, 4.9],
    [2.1, 1.7, 4.8],
    [2.6, 4.5, 4.4],
    [5.0, 5.8, 1.6],
    [4.4, 5.5, 1.4],
    [1.8, 1.1, 0.4],
    [2.3, 0.5, 0.6],
    [0.7, 4.2, 3.9],
    [1.4, 4.4, 3.5],
    [3.0, 1.9, 1.0],
    [2.6, 2.5, 1.2],
    [0.9, 5.2, 4.9],
    [2.3, 5.8, 5.8],
    [2.2, 4.5, 1.8],
    [4.5, 0.1, 3.0],
    [3.0, 0.9, 5.1],
    [3.1, 0.5, 5.8],
    [4.8, 2.7, 5.4],
    [5.3, 3.1, 5.7],
    [0.1, 4.9, 2.9],
    [2.9, 5.0, 3.7],
  ])
  pairs_ref = np.array([
    [ 0, 45],
    [ 1,  2],
    [ 3, 13],
    [ 4,  5],
    [ 6, 40],
    [ 7, 36],
    [ 8,  9],
    [10, 11],
    [12, 24],
    [14, 19],
    [15, 25],
    [16, 17],
    [18, 86],
    [20, 37],
    [21, 59],
    [22, 23],
    [26, 27],
    [28, 58],
    [29, 88],
    [30, 31],
    [33, 70],
    [34, 69],
    [35, 44],
    [38, 39],
    [41, 89],
    [42, 43],
    [46, 47],
    [48, 49],
    [50, 51],
    [52, 53],
    [54, 55],
    [56, 57],
    [60, 61],
    [62, 63],
    [64, 65],
    [66, 87],
    [67, 76],
    [68, 94],
    [72, 73],
    [74, 75],
    [77, 95],
    [78, 79],
    [80, 81],
    [82, 83],
    [84, 85],
    [90, 91],
    [92, 93],
  ])
  rmax = 1.0
  rij = axes_pos.auto_distance_table(axes, pos)
  pairs = axes_pos.find_dimers(rij, rmax)
  assert np.allclose(pairs, pairs_ref)

#if __name__ == '__main__':
#  test_find_dimers()
## end __main__
