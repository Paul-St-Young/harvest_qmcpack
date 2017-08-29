import os
import lxml.etree as etree

def read(fname):
  parser = etree.XMLParser(remove_blank_text=True)
  doc    = etree.parse(fname,parser)
  return doc

def write(fname,doc):
  doc.write(fname,pretty_print=True)

def show(node):
  print( etree.tostring(node,pretty_print=True) )

def swap_in_opt_wf(inp_fname,iqmc):
 
  opt_dir = os.path.dirname(inp_fname)

  # read the optimization input for nqmc & prefix to find .opt files
  doc    = read(inp_fname)
  nqmc   = int( doc.find('.//loop').get('max') )
  assert iqmc < nqmc

  prefix = doc.find('.//project').get('id')
  stext  = 's'+str(iqmc).zfill(3)
  fopt   = '.'.join([prefix,stext,'opt','xml'])

  # read optimized wavefunction
  doc1 = read(fopt)

  wf1 = doc1.find('.//wavefunction')
  wf0 = doc.find('.//wavefunction')
  wup = wf0.getparent()
  idx = wup.index(wf0)
  wup.remove(wf0)
  wup.insert(idx,wf1)

  return doc
# end def
