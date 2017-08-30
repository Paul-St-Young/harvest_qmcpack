from qharv.seed import xml

def lattice_vectors(fname):
  doc = xml.read(fname)
  sc_node = doc.find('.//simulationcell')
  lat_node = sc_node.find('.//parameter[@name="lattice"]')
  unit = lat_node.get('units')
  assert unit == 'bohr'
  axes = xml.text2arr( lat_node.text )
  return axes

def atomic_coords(fname,pset_name='ion0'):
  # !!!! assuming atomic units (bohr)
  # !!!! finds the first group in particleset
  doc = xml.read(fname)
  source_pset_node = doc.find('.//particleset[@name="%s"]'%pset_name)
  pos_node = source_pset_node.find('.//attrib[@name="position"]')
  pos = xml.text2arr(pos_node.text)
  return pos
