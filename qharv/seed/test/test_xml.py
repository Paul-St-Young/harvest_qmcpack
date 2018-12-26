from qharv.seed import xml

def test_parse():
  text = '<root/>'
  node = xml.parse(text)
  assert node.tag == 'root'

def test_read():
  text = '<root/>'
  fname = 'scratch_read.xml'
  with open(fname, 'w') as f:
    f.write(text)
  doc = xml.read(fname)
  root = doc.getroot()
  assert root.tag == 'root'

def test_write():
  text = '<root> <leaf/> </root>'
  fname = 'scratch_write.xml'
  node = xml.parse(text)
  doc = xml.etree.ElementTree(node)
  xml.write(fname, doc)
  with open(fname, 'r') as f:
    text1 = f.read()
  assert text1 == '<root>\n  <leaf/>\n</root>\n'

def test_ls():
  text = '<root> <leaf1/> <leaf2><child/></leaf2> </root>'
  node = xml.parse(text)
  mystr = xml.ls(node)
  assert mystr == 'leaf1\nleaf2\n'
def test_lsr():
  text = '<root> <leaf1/> <leaf2><child/></leaf2> </root>'
  node = xml.parse(text)
  mystr = xml.ls(node, r=1)
  assert mystr == 'leaf1\nleaf2\n  child\n'

def test_get_nelec():
  text = '''<root><particleset name="e">
    <group name="u" size="5"/>
  </particleset></root>'''
  node = xml.parse(text)
  assert xml.get_nelec(node) == 5
  text = '''<root><particleset name="e">
    <group name="u" size="5"/>
    <group name="d" size="7"/>
  </particleset></root>'''
  node = xml.parse(text)
  assert xml.get_nelec(node) == 12
  text = '''<root><particleset name="e">
    <group name="u" size="5"/>
    <group name="d" size="7"/>
    <group name="p" size="7"/>
  </particleset></root>'''
  node = xml.parse(text)
  assert xml.get_nelec(node) == 12
