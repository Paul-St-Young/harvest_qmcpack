#!/usr/bin/env python
import numpy as np
from qharv.inspect import axes_pos

axes0 = np.array([
  [2.46637878,-4.27189335, 0.        ],
  [2.46637878, 4.27189335, 0.        ],
  [0.        , 0.        , 8.45173341]
])

pos0 = np.array([
  [ 1.56156541,  1.57330776,  7.42305655],
  [ 1.56156506,  2.69858287,  3.19719163],
  [ 3.37119214, -2.69858455,  5.25453794]
])

dists0 = [4.37311998,2.26578821,2.43520807]

bcc0 = 0.5*np.array([
  [-1.0, 1.0, 1.0],
  [ 1.0,-1.0, 1.0],
  [ 1.0, 1.0,-1.0],
])

fcc0 = 0.5*np.array([
  [ 0.0, 1.0, 1.0],
  [ 1.0, 0.0, 1.0],
  [ 1.0, 1.0, 0.0],
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
  disp_table = [axes_pos.displacement(axes0,pos0[i],pos0[j]) for (i,j) in combinations(range(nptcl),2)]
  dists = np.linalg.norm(disp_table,axis=1)
  assert np.allclose(dists,dists0)
# end def

def test_auto_distance_table():
  dtable = axes_pos.auto_distance_table(axes0,pos0)

  nptcl  = len(pos0)
  i_triu = np.triu_indices(nptcl,m=nptcl,k=1)
  dists  = dtable[i_triu]
  assert np.allclose(dists,dists0)
# end def

def test_volume():
  # primitive cell of b.c.c. with lattice constant 1.0
  assert np.allclose(bcc0, 0.5*(np.ones(3)-2.*np.eye(3)))
  # primitive cell of f.c.c. with lattice constant 1.0
  assert np.allclose(fcc0, 0.5*(np.ones(3)-1.*np.eye(3)))
  # 2 atoms in b.c.c. conventional cell
  assert np.isclose( axes_pos.volume(bcc0), 0.5)
  # 4 atoms in f.c.c. conventional cell
  assert np.isclose( axes_pos.volume(fcc0), 0.25)
  # no negative volume
  axes_neg = np.array([
    [  2.55252043e+00,   9.10248160e+00,  -4.45451981e+00],
    [  5.15835511e+00,   1.16509760e-03,  -8.90903961e+00],
    [  5.15835511e+00,   1.16509760e-03,   8.90903961e+00]
  ])
  assert np.isclose( axes_pos.volume(axes_neg), 836.574116485)
# end def

def test_raxes():
  b2f = axes_pos.raxes(bcc0)/(2.*np.pi)/2.
  assert np.allclose(b2f,fcc0)
  f2b = axes_pos.raxes(fcc0)/(2.*np.pi)/2.
  assert np.allclose(f2b,bcc0)
# end def

def test_rwsc():
  # a fairly extreme slab-like cell
  axes = np.array([
   [ 7.73625309, 18.39193864, -13.47014157],
   [-0.08306451, 27.58607338,  17.96016072],
   [-2.59722820, -0.00040676,  13.47012604]
  ])
  # dn = 2 is sufficient !!!! THIS IS WRONG !!!!
  rwsc = axes_pos.rwsc(axes, dn=2)
  assert np.isclose(rwsc, 5.281986)
# end def

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
