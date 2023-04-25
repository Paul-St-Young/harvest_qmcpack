from qharv.seed import xml, qmcpack_in

def read_example(fxml):
  import os
  path = os.path.dirname(os.path.realpath(__file__))
  fxml = os.path.join(path, fxml)
  doc = xml.read(fxml)
  return doc

def test_meta_from_parameters():
  fxml = 'examples/eg3d-nup7-ndn0-dmc.xml'
  doc = read_example(fxml)
  vmc = doc.find('.//qmc[@method="vmc"]')
  meta = qmcpack_in.meta_from_parameters(vmc)
  expect = dict(
    method = 'vmc',
    move = 'pbyp',
    checkpoint = '-1',
    blocks_between_recompute = '10',
    blocks = '400',
    warmupsteps = '100',
    timestep = '0.5',
    subSteps = '1',
    steps = '2',
    samples = '2304',
  )
  for key, val0 in expect.items():
    val1 = meta.pop(key)
    assert val0 == val1
  assert len(meta) == 0

def test_output_prefix_meta():
  fxml = 'examples/eg3d-nup7-ndn0-dmc.xml'
  doc = read_example(fxml)
  pm = qmcpack_in.output_prefix_meta(doc)
  pres = sorted(pm.keys())
  expect = ['dmc.s000', 'dmc.s001', 'dmc.s002']
  assert pres == expect
  pm = qmcpack_in.output_prefix_meta(doc, group=2)
  pres = sorted(pm.keys())
  expect = ['dmc.g002.s000', 'dmc.g002.s001', 'dmc.g002.s002']
  assert pres == expect

def test_random_twists():
  import numpy as np
  twists = qmcpack_in.random_twists(3, ndim=2, method='Halton', center=True)
  refs = [
    [0.  ,       0.        ],
    [-0.5 ,      0.33333333],
    [0.25,      -0.33333333]
  ]
  assert np.allclose(twists, refs)
  twists = qmcpack_in.random_twists(2, ndim=3, method='Sobol', center=True)
  refs = [
    [0, 0, 0],
    [-0.5, -0.5, -0.5],
  ]
  assert np.allclose(twists, refs)
  twists = qmcpack_in.random_twists(2, ndim=3, method='Sobol',
    nskip=8, nevery=4)
  refs = [
    [0.1875, 0.3125, 0.9375],
    [0.3125, 0.1875, 0.3125]
  ]
  assert np.allclose(twists, refs)
