# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to visualize jastrow potential U. Jastrow J=exp(-U).
#
# Most function take lxml.etree.Element as first input (doc). For example:
#  doc = etree.parse(fxml)
#  doc1 = doc.find('.//wavefunction')
import numpy as np

from qharv.seed import xml
from qharv.reel import mole


def find_iopt(iopt, opt_dir):
  """ locate opt.xml file in opt_dir
  Args:
    iopt (int): optimization step (series id)
    opt_dir (str): path to optimization run
  Return:
    str: floc, location of the requested opt.xml file
  """
  st = 's'+str(iopt).zfill(3)
  regex = '*%s.opt.xml' % st
  flist = mole.files_with_regex(regex, opt_dir)
  if len(flist) != 1:
    raise RuntimeError('find %d expected 1' % len(flist))
  floc = flist[0]
  return floc


def get_coeff(doc, coef_id):
  """ extract coefficients from a <coefficients> node

  Args:
    doc (etree.Element): <coefficient> node
    coef_id (str): coefficient name (id), e.g. 'uu', 'ud', 'cG2'
  Return:
    array-like: extracted coefficients
  """
  cnode = doc.find('.//coefficients[@id="%s"]' % coef_id)
  coefs = np.array(cnode.text.split(), dtype=float)
  return coefs
