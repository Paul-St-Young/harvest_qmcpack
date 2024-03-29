# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to parse ASCII output. Mostly built around mmap's API.
#  The central object is mmap.mmap, which is usually named "mm".
from mmap import mmap

def read(fname):
  """ get a memory map pointer to file

  Args:
    fname (str): filename
  Return:
    mmap.mmap: memory map to file
  """
  with open(fname, 'r+') as f:
    mm = mmap(f.fileno(), 0)
  return mm

def stay(read_func, *args, **kwargs):
  """ stay at current memory location after read

  Args:
    Callable: read function, which takes mmap as first input
  Return:
    Callable: read but no change to mmap tell()
  """
  def wrapper(mm, *args, **kwargs):
    idx = mm.tell()
    ret = read_func(mm, *args, **kwargs)
    mm.seek(idx)
    return ret
  return wrapper

@stay
def get_key_value_pairs(mm, sep='='):
  """ read all key value pairs using separator

  Args:
    mm (mmap.mmap): memory map
  Return:
    dict: string->string key-value pairs
  """
  idxl = all_lines_with_tag(mm, sep)
  entry = {}
  for idx in idxl:
    mm.seek(idx)
    ibegin = mm.rfind(b'\n', 0, idx)
    mm.seek(ibegin+1)  # rewind to beginning of line
    line = mm.readline()
    tokens = line.split(sep.encode())
    name = tokens[0].strip()  # strip whitespace
    val = tokens[1].strip()
    entry[name.decode()] = val.decode()
  return entry

@stay
def name_sep_val(mm, name, sep='=', dtype=float, pos=1, ival=0):
  """ read key-value pair such as "name = value"
  e.g.
  name_sep_val(mm, 'a'): 'a = 2.4'
  name_sep_val(mm, 'volume', pos=-2): 'volume = 100.0 bohr^3'
  name_sep_val(mm, 'key', sep=':'): 'key:val'
  name_sep_val(mm, 'new', sep=':'): 'new:name'
  name_sep_val(mm, 'natom', dtype=int): 'new:name'

  Args:
    mm (mmap.mmap): memory map
    name (str): name of variable; used to find value line
    sep (str, optional): separator, default '='
    dtype (type, optional): variable data type, default float
    pos (int, optional): position of value in line, default 1 (after sep)
    ival (int, optional): index of chosen value in line, default 0 (first)
  Return:
    dtype: value of requested variable
  """
  idx = mm.find(name.encode())
  if idx == -1:
    raise RuntimeError(name+' not found')
  mm.seek(idx)
  line = mm.readline().decode()
  tokens = line.split(sep)

  # assume the text immediately next to the separator is the desired value
  val_text = tokens[pos] if ival is None else tokens[pos].split()[ival]
  val = dtype(val_text)
  return val

@stay
def all_lines_with_tag(mm, tag, nline_max=1024*1024):
  """ return a list of memory indices pointing to the start of tag
   the search is conducted starting from the current location of mm.

  Args:
    mm (mmap.mmap): memory map to file
    tag (str): tag to identify lines to look for
    nline_max (int, optional): maximum number of lines to look for
     , default is 2^20. Error will be raised if max is too low
  Return:
    list: a list of memory locations of all found tags
  """
  all_idx = []
  for iline in range(nline_max):
    idx = mm.find(tag.encode())
    if idx == -1:
      break
    mm.seek(idx)
    all_idx.append(idx)
    mm.readline()

  # guard
  if iline >= nline_max-1:
    raise RuntimeError('may need to increase nline_max')
  return all_idx

@stay
def all_lines_at_idx(mm, idx_list):
  """ return a list of lines given a list of memory locations
  follow up on all_lines_with_tag
  e.g. all_lines_at_idx(mm, all_lines_with_tag(mm, 'Atom') )
  reads '''
  Atom 0 0 0 0
  Atom 1 1 1 1
  Atom 2 2 2 2
  '''

  Args:
    mm (mmap.mmap): memory map to file
    idx_list (list): a list of memory locations (int)
  Return:
    list: a list of strings, each being the line at idx
  """
  lines = []
  for idx in idx_list:
    mm.seek(idx)
    # row back to beginning of line
    ibegin = mm.rfind(b'\n', 0, idx)
    if ibegin == -1:
      ibegin = 0
    mm.seek(ibegin)
    mm.readline()
    # read desired line
    line = mm.readline()
    lines.append(line.decode())
  return lines

@stay
def locate_block(mm, header, trailer, force_head=False, force_tail=False,
                 skip_header=True, skip_trailer=None):
  """ find the memory locations bounding a block of text
  in between header and trailer; header and trailer are
  not included by default
  e.g. see block_text

  Args:
    mm (mmap.mmap): memory map to text file
    header  (str): string indicating the beginning of block
    trailer (str): string indicating the end of block
    skip_head (bool, optional): skip header, default is True
    skip_trailer (bool, optional): skip trailer, default is True,
      unless force_tail, then default to False
  Return:
    tuple: (begin_idx, end_idx), memory span of text block
  """
  if skip_trailer is None:
    skip_trailer = not force_tail
  begin_idx = mm.find(header.encode())
  if begin_idx == -1:
    if force_head:
      begin_idx = 0
    else:
      raise RuntimeError('failed to find "%s"' % header)
  if skip_header:
    mm.seek(begin_idx)
    mm.readline()
    begin_idx = mm.tell()
  end_idx = mm.find(trailer.encode())
  whence = 1  # seek from current location by default
  if end_idx == -1:
    if force_tail:
      whence = 2  # seek from end of file
    else:
      raise RuntimeError('failed to find "%s"' % trailer)
  if not skip_trailer:
    mm.seek(end_idx, whence)
    mm.readline()
    end_idx = mm.tell()
  return begin_idx, end_idx

def block_text(mm, header, trailer, **kwargs):
  """ find a block of text in between header and trailer
  header and trailer are not included by default

  e.g. given text in mm
  '''
  begin important data
  1 2 3
  4 5 6
  7 8 9
  end important data
  '''

  mm.block_text(mm, 'begin', 'end') returns
  '''1 2 3
  4 5 6
  7 8 9
  '''

  Args:
    mm (mmap.mmap): memory map to text file
    header  (str): string indicating the beginning of block
    trailer (str): string indicating the end of block
  """
  bidx, eidx = locate_block(mm, header, trailer, **kwargs)
  return mm[bidx:eidx].decode()

def lr_mark(line, lmark, rmark):
  """ read a string segment from line, which is enclosed between l&rmark
   e.g. extract the contents in parenteses
  Args:
    line (str): text line
    lmark (str): left marker, e.g. '('
    rmark (str): right marker, e.g. ')'
  Return:
    str: text in between left and right markers
  """
  lidx = line.find(lmark)
  assert lidx != -1
  ridx = line.find(rmark)
  assert ridx != -1
  return line[lidx+1:ridx]

def name_val_table(text, dtype=float):
  """ designed to parse optVariables text block
  e.g. '''uu_0 1.0770e+00 1 1  ON 0
uu_1 6.7940e-01 1 1  ON 1
uu_2 4.3156e-01 1 1  ON 2
ud_0 1.6913e+00 1 1  ON 5
ud_1 1.0443e+00 1 1  ON 6
ud_2 6.1912e-01 1 1  ON 7
'''

  return variable-value map, only the first two columns are parsed.

  Args:
    text (str): text block such as given in the example
    dtype (type): data type of value, default is float
  Return:
    dict: variable name -> value map
  """
  lines  = text.split('\n')[:-1]

  var_dict = {}
  for line in lines:
    tokens = line.split()
    name = tokens[0]
    val  = dtype(tokens[1])
    var_dict[name] = val
  return var_dict

def change_line(text, t0, t1):
  text1 = ''
  for line in text.split('\n'):
    if t0 in line:
      text1 += t1
    else:
      text1 += line
    text1 += '\n'
  return text1.strip('\n')
