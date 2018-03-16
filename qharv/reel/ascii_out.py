# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse ASCII output. Mostly built around mmap's API.

from mmap import mmap
import pandas as pd

def read(fname):
  with open(fname,'r+') as f:
    mm = mmap(f.fileno(),0)
  # end with
  return mm

def name_sep_val(mm,name,sep='=',val_dtype=float,pos=-1):
  idx = mm.find(name)
  if idx == -1:
    raise RuntimeError(name+' not found')
  mm.seek(idx)
  line = mm.readline()
  tokens = line.split(sep)

  # assume the text immediately next to the separator is the desired value
  val_text = tokens[pos].split()[0]
  val = val_dtype( val_text )
  return val
# end def

def all_lines_with_tag(mm,tag,nline_max=1024*1024):
  """ return a list of memory indices pointing to the start of tag
   the search is conducted starting from the current location of mm. """
  all_idx = []
  for iline in range(nline_max):
    idx = mm.find(tag)
    if idx == -1:
      break
    # end if
    mm.seek(idx)
    all_idx.append(idx)
    mm.readline()
  # end for iline

  # guard
  if iline >= nline_max-1:
    raise RuntimeError('may need to increase nline_max')
  # end if
  return all_idx
# end def all_lines_with_tag

def all_lines_at_idx(mm,idx_list):
  lines = []
  for idx in idx_list:
    mm.seek(idx)
    lines.append( mm.readline() )
  # end for
  return lines
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

def lr_mark(line,lmark,rmark):
  """ read a string segment from line, which is enclosed between l&rmark
   e.g. extract the contents in parenteses
  """
  lidx = line.find(lmark)
  assert lidx != -1
  ridx = line.find(rmark)
  assert ridx != -1
  return line[lidx+1:ridx]
# end def

def parse_qmcas_output(fname):
  """ !!!! hacky function to gather QMCPACK scalar.dat data !!!! not recommended for general use !!!! 
   problems: 
    1. input file is not known, cannot assign method to series number 'iqmc'
    2. equilibration length is not known """
  # parse outputs from "qmca -q ev --sac */*scalar.dat" such as:
  #  ./detsci49_af/c2  series 0  -10.452004 +/- 0.006424    1.6   1.601253 +/- 0.068660    1.1   0.1532

  data = {'filepath':[],'iqmc':[],'LocalEnergy_mean':[],'LocalEnergy_error':[],'Variance_mean':[],'Variance_error':[],'correlation':[]}
  with open(fname) as f:
    for line in f:
      tokens = line.split()
      if len(tokens) == 12:
        filepath = tokens[0]

        # determine method
        iqmc = int(tokens[2])

        # read energy
        energy_mean  = float(tokens[3])
        energy_error = float(tokens[5])

        # read correlation
        correlation  = float(tokens[6])

        # read variance
        variance_mean  = float(tokens[7])
        variance_error = float(tokens[9])

        # organize data in table
        label_and_data = zip(
          ['filepath','LocalEnergy_mean','LocalEnergy_error','iqmc','Variance_mean','Variance_error','correlation'],
          [filepath,energy_mean,energy_error,iqmc,variance_mean,variance_error,correlation]
        )
        for key,val in label_and_data:
          data[key].append(val)
        # end for
      # end if
    # end for 
  # end with
  mydf = pd.DataFrame(data)
  return mydf
# end def parse_qmcas_output
