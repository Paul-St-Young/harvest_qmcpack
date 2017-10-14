import os
import urllib
import numpy as np

fname = 'TIP5P_PIMC.32.P5.0C.0.ptcl.xml'

def save_xml_example():
  if not os.path.isfile(fname):
    flink = 'https://sites.google.com/a/cmscc.org/qmcpack/developers-forum/input-xml-schema/TIP5P_PIMC.32.P5.0C.0.ptcl.xml'
    response = urllib.urlopen(flink)
    text = response.read()
    with open(fname,'w') as fp:
      fp.write(text)
    # end with
  # end if
# end def

def test_axes():
  save_xml_example()
  from qharv.inspect.crystal import lattice_vectors
  axes = lattice_vectors(fname)
  assert np.allclose(axes,18.330056710775050*np.eye(3))
# end def test_axes
