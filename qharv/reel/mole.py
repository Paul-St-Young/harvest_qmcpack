# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# dig around for goodies
import subprocess as sp

def findall(regex, rundir, prepend_star=True, **kwargs):
  """ find all matching files with regular expression in rundir

  Args:
    regex (str):  regular expression for file names
    rundir (str): directory containing the files to be found
    prepend_star (bool, optional): add '*' to regex, default True
  Return:
    list: flist, list of all matching files
  """
  if prepend_star:
    myregex = '*'+regex
  else:
    myregex = regex
  flist = files_with_regex(myregex, rundir, **kwargs)
  return flist

def files_with_regex(regex, rundir, case=True, ftype='f', **kwargs):
  """ find files with the given suffix in folder rundir
  rely on bash `find` command

  Args:
    regex (str):  regular expression for file names
    rundir (str): directory containing the files to be found
    case (bool, optional): case sensity, default is True
    ftype (str, optional): files type, default is regular file 'f'
    , may be directory 'd'
  Return:
    list: flist, a list of filenames matching the given regular expression
  """
  popt = '-path'
  if not case:
    popt = '-ipath'  # not case sensitive
  options = []
  for key, val in kwargs.items():
    options.append('-'+key)
    options.append(str(val))
  cmdl = ['find', rundir] + options + [popt, regex, '-type', ftype]
  out = sp.check_output(cmdl)
  flist = out.decode().split('\n')[:-1]
  return flist

def find(regex, rundir, **kwargs):
  """ find the first file that matches the given regular expression
  RuntimeError will be raised unless exactly one file is found

  Args:
    regex (str):  regular expression for file names
    rundir (str): directory containing the files to be found
  Return:
    str: filename
  """
  flist = files_with_regex(regex, rundir, **kwargs)
  if len(flist) != 1:
    raise RuntimeError('expect 1 but found %d' % len(flist))
  return flist[0]

def clean_path(path):
  """ remove . and .. from path

  Args:
    path (str): file or folder path
  Return:
    str: clean path
  """
  segs = path.split('/')
  segs1 = [seg for seg in segs if seg not in ['.', '..']]
  path1 = '/'.join(segs1)
  return path1

def interpret_qmcpack_fname(fname):
  """ extract metadata regarding the contents of a file based on its filename.
  QMCPACK generates files having a pre-determined suffix structure. This
  function will interpret the last 4 period-separated segments of the suffix.

  fname examples:
    qmc.s000.scalar.dat
    qmc.g000.s000.stat.h5
    qmc.g161.s000.config.h5
    qmc.g005.s001.cont.xml

  Args:
    fname (str): filename, must end in one of ['dat','h5','qmc','xml'].
  Return:
    dict: a dictionary of metadata.
  """
  known_extensions = set(['dat', 'h5', 'qmc', 'xml'])

  tokens = fname.split('.')
  ext = tokens[-1]  # dat,h5,qmc
  if ext not in known_extensions:
    raise RuntimeError('unable to interpret %s' % fname)
  # end if

  # interpret various pieces of the filename

  # category
  cate   = tokens[-2]  # scalar,stat,config,random,qmc

  # series index
  isst   = tokens[-3]  # s000
  iss    = int(isst.replace('s', ''))  # series index

  # group index
  grouped = False  # single input is not grouped
  igt = tokens[-4]  # g000 or $prefix
  ig = 0  # group index
  suf_list = [isst, cate, ext]
  if igt.startswith('g') and len(igt) == 4:
    ig = int(igt.replace('g', ''))
    suf_list = [igt] + suf_list
    grouped = True
  else:  # there is no group index
    pass  # keep defaul ig=0, grouped=False
  # end if

  # get project id by removing the suffix
  suffix = '.' + '.'.join(suf_list)
  prefix = fname.replace(suffix, '')

  # metadata entry
  entry = {
    'id': prefix, 'group': ig, 'series': iss,
    'category': cate, 'ext': ext, 'grouped': grouped
  }
  return entry

def build_qmcpack_fname(entry):
  """ inverse of interpret_qmcpack_fname

  Args:
    entry (dict): a dictionary of meta data, must include
    ['id','grouped','group','series','category','ext'] in key
  Return:
    str: filename
  """
  order = ['id', 'series', 'category', 'ext']
  if entry['grouped']:
    order.insert(1, 'group')
  # end if
  tokens = []
  for key in order:
    val = entry[key]
    # get string representation
    if key == 'group':
      val = 'g'+str(val).zfill(3)
    elif key == 'series':
      val = 's'+str(val).zfill(3)
    else:
      val = str(val)
    # end if
    tokens.append(val)
  # end for
  fname = '.'.join(tokens)
  return fname
