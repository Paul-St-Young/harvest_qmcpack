# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse ASCII output. Mostly built around mmap's API.

from mmap import mmap

def read(fname):
  with open(fname,'r+') as f:
    mm = mmap(f.fileno(),0)
  # end with
  return mm

def name_sep_val(mm,name,sep='=',val_dtype=float):
  idx = mm.find(name)
  if idx == -1:
    raise RuntimeError(name+' not found')
  mm.seek(idx)
  line = mm.readline()
  tokens = line.split(sep)

  # assume the text immediately next to the separator is the desired value
  val_text = tokens[1].split()[0]
  val = val_dtype( val_text )
  return val
# end def
