# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse QMCPACK ASCII output.
from qharv.reel import ascii_out

def opt_vars(fout, begin_tag='<optVariables', end_tag='</optVariables'):
  """Parse optimizable variables

  Args:
    fout (str): optimization output
    begin_tag (str, optional): default '<optVariables'
    end_tag (str, optional): default '</optVariables'
  Return:
    list: a list of dictionary, each a key-value map of opt. var.s
  """
  mm = ascii_out.read(fout)
  idxl = ascii_out.all_lines_with_tag(mm, begin_tag)
  entryl = []
  for iopt, idx in enumerate(idxl):
    mm.seek(idx)
    text = ascii_out.block_text(mm, begin_tag, end_tag)
    entry = ascii_out.name_val_table(text)
    entryl.append(entry)
  return entryl

def orb_rot_meta(name):
  """Parse metadata from orbital rotation variable name

  Args:
    name (str): optimizable variable name
  Return:
    dict: metadata
  Example:
    >>> name = "spo-up_orb_rot_0000_0002"
    >>> orb_rot_meta(name)
    >>> {'prefix': 'spo-up', 'i': 0, 'j': 2}
  """
  useful = name.replace('orb_rot_', '')
  tokens = useful.split('_')
  i = int(tokens[-2])
  j = int(tokens[-1])
  suffix = '_%04d_%04d' % (i, j)
  prefix = useful.replace(suffix, '')
  meta = {'prefix': prefix, 'i': i, 'j': j}
  return meta

def orb_rot_opt_vars(fout):
  """Build a database of orbital rotation parameters

  Args:
    fout (str): optimization output
  Return:
    pd.DataFrame: database of orbital rotation parameters
  """
  import pandas as pd
  entryl = opt_vars(fout)
  data = []
  for iopt, entry in enumerate(entryl):
    for name, value in entry.items():
      meta = orb_rot_meta(name)
      meta.update({'value': value})
      meta['iopt'] = iopt
      data.append(meta)
  df = pd.DataFrame(data)
  return df
