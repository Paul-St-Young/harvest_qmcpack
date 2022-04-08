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
