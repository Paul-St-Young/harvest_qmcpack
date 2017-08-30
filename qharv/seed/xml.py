# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate an xml input. Almost all functions are built around the lxml module's API.
import os
import numpy as np
from lxml import etree

def read(fname):
  parser = etree.XMLParser(remove_blank_text=True)
  doc    = etree.parse(fname,parser)
  return doc

def write(fname,doc):
  doc.write(fname,pretty_print=True)

def show(node):
  print( etree.tostring(node,pretty_print=True) )

def arr2text(arr):
  """ format a numpy array into a text string """
  text = ''
  if len(arr.shape) == 1: # vector
      text = " ".join(arr.astype(str))
  elif len(arr.shape) == 2: # matrix
      mat  = [arr2text(line) for line in arr]
      text = "\n" + "\n".join(mat) + "\n"
  else:
      raise RuntimeError('arr2text can only convert vector or matrix.')
  # end if
  return text
# end def arr2text

def text2arr(text,dtype=float,flatten=False):
  """ convert a text string into a numpy array """
  tlist = text.strip(' ').strip('\n').split('\n')
  if len(tlist) == 1:
    return np.array(tlist,dtype=dtype)
  else:
    if flatten:
      mytext = '\n'.join(['\n'.join(line.split()) for line in tlist])
      myarr = text2arr(mytext)
      return myarr.flatten()
    else:
      return np.array([line.split() for line in tlist],dtype=dtype)
    # end if
  # end if
# end def text2arr

def opt_wf_fname(opt_inp,iqmc):
  """ Find the file containing the optimized <wavefunction> at optimization loop iqmc 

  example of a folder containing an optimization run:
  $ls opt_dir
    opt.xml
    qmc.s000.scalar.dat
    qmc.s000.opt.xml
    qmc.s001.scalar.dat
    qmc.s001.opt.xml
  $
  opt_wf_fname('opt_dir/opt.xml',1) returns 'opt_dir/qmc.s001.opt.xml'

  Args:
    opt_inp (str): optimization run input file
    iqmc (int): optimization loop to target
  Returns:
    str: wf_fname, name of the xml file containing the optimized <wavefunction>
  """

  # read the optimization input for nqmc & prefix to find .opt files
  doc = read(opt_inp)
  nqmc   = int( doc.find('.//loop').get('max') )
  assert iqmc < nqmc

  # read project prefix to determine .opt filename
  prefix = doc.find('.//project').get('id')
  stext  = 's'+str(iqmc).zfill(3)
  fopt   = '.'.join([prefix,stext,'opt','xml'])

  # return location of file
  opt_dir  = os.path.dirname(opt_inp)
  wf_fname = os.path.join(opt_dir,fopt)

  return wf_fname
# end def opt_wf_fname

def swap_in_opt_wf(inp_fname,wf_fname):
  """ Put an optimized wavefunction into an xml input 

  Designed to help continue a wavefunction optimization. One can also use optimized wavefunction in a VMC or DMC calculation, but the <loop> section will have to be removed, and new <qmc> sections added. See xml_examples.py.

  Args:
    inp_fname (str): xml input file having an old <wavefunction>
    wf_fname  (str): xml file containing the optimized <wavefunction>
  Returns:
    lxml.etree._ElementTree: xml input with optimized wavefunction """

  # find <wavefunction>
  doc  = read(inp_fname)
  doc1 = read(wf_fname)
  wf1 = doc1.find('.//wavefunction')
  wf0 = doc.find('.//wavefunction')

  # swap <wavefunction>
  wup = wf0.getparent()
  idx = wup.index(wf0)
  wup.remove(wf0)
  wup.insert(idx,wf1)

  return doc
# end def swap_in_opt_wf
