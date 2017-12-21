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
