# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse QMCPACK ASCII output.
from qharv.reel import ascii_out

def opt_vars(fout, begin_tag='<optVariables', end_tag='</optVariables'):
  mm = ascii_out.read(fout)
  idxl = ascii_out.all_lines_with_tag(mm, begin_tag)
  entryl = []
  for iopt, idx in enumerate(idxl):
    mm.seek(idx)
    text = ascii_out.block_text(mm, begin_tag, end_tag)
    entry = ascii_out.name_val_table(text)
    entryl.append(entry)
  return entryl
