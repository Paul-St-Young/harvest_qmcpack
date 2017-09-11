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

def locate_block(mm,header,trailer,skip_header=True,skip_trailer=True):
    begin_idx = mm.find(header.encode())
    if skip_header:
      mm.seek(begin_idx)
      mm.readline()
      begin_idx = mm.tell()
    # end if
    end_idx   = mm.find(trailer.encode())
    if not skip_trailer:
      mm.seek(end_idx)
      mm.readline()
      end_idx = mm.tell()
    # end if
    return begin_idx,end_idx
# end def locate_block

def block_text(mm,header,trailer,skip_header=True,skip_trailer=True):
    bidx,eidx = locate_block(mm,header,trailer)
    return mm[bidx:eidx]
# end def block_text
