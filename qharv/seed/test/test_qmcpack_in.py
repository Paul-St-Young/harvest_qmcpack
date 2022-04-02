from qharv.seed import xml, qmcpack_in

def test_meta_from_parameters():
  import os
  path = os.path.dirname(os.path.realpath(__file__))
  fxml = os.path.join(path, 'examples/eg3d-nup7-ndn0-dmc.xml')
  print(fxml)
  doc = xml.read(fxml)
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
    print(key, val0, val1)
    assert val0 == val1
  assert len(meta) == 0
