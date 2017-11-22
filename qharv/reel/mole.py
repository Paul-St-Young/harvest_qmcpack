# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# dig around for goodies
import os
import subprocess as sp
from lxml import etree

def files_with_regex(regex,rundir,case=True,ftype='f'):
  """ find files with the given suffix in folder rundir; rely on bash `find` command
  Args:
    regex (str):  regular expression for file names
    rundir (str): directory containing the files to be found
    case (bool, optional): case sensity, default is True
    ftype (str, optional): files type, default is regular file 'f', may be directory 'd'
  Returns:
    list: flist, a list of filenames matching the given regular expression
  """
  popt = '-path'
  if not case:
    popt = '-ipath' # not case sensitive
  out = sp.check_output(['find',rundir,popt,regex,'-type',ftype])
  flist = out.split('\n')[:-1]
  return flist
# end def files_with_regex
def files_scalar_dat(calcdir):
  return files_with_regex('*scalar.dat',calcdir)
def files_stat_h5(calcdir):
  return files_with_regex('*stat.h5',calcdir)

def group_map(flist):
  """ group QMCPACK output files by series id; if a file does not have a group index, then append it to an indepenednt list
  Args:
    flist (list): a list of filenames, should be QMCPACK outputs
  Returns:
    dict: groups, map series id (int) to a list of filenames
    list: indep, a list of independent filenames
  """
  indep  = []
  groups = {}
  for floc in flist:
    fname  = os.path.basename(floc)
    tokens = fname.split('.')
    if tokens[-4].startswith('g'):
      # e.g. dmc.g006.s000.scalar.dat
      st = tokens[-3]
      iss= int(st.replace('s',''))
      if iss in groups.keys():
        groups[iss].append(fname)
      else:
        groups[iss] = [fname]
      # end if
    else:
      # e.g. dmc.s000.scalar.dat
      indep.append(fname)
    # end if
  # end for
  return groups,indep
# def group_map

def interpret_qmcpack_fname(fname):
  """ extract metadata regarding the contents of a file based on its filename. QMCPACK generates files having a pre-determined suffix structure. This function will interpret the last 4 period-separated segments of the suffix. 
  Args:
    fname (str): filename, must end in one of ['dat','h5','qmc','xml'].
  Returns:
    dict: a dictionary of metadata. 
  """
  known_extensions = set( ['dat','h5','qmc','xml'] )

  tokens = fname.split('.')
  ext = tokens[-1] # dat,h5,qmc
  if ext not in known_extensions:
    raise RuntimeError('unable to interpret %s' % fname)
  # end if

  # interpret various pieces of the filename

  # category
  cate   = tokens[-2] # scalar,stat,config,random,qmc

  # series index
  isst   = tokens[-3] # s000
  iss    = int(isst.replace('s','')) # series index ('is' is a Python keyword)

  # group index
  igt    = tokens[-4] # g000 or $prefix
  ig = 0 # group index
  suf_list = [isst,cate,ext]
  if igt.startswith('g') and len(igt)==4:
    ig = int(igt.replace('g',''))
    suf_list = [igt] + suf_list
  else:  # there is no group index
    pass # keep defaul ig=0
  # end if

  # get project id by removing the suffix
  suffix = '.' + '.'.join(suf_list)
  prefix = fname.replace(suffix,'')

  # metadata entry
  entry = {'id':prefix,'group':ig,'series':iss,'category':cate,'ext':ext}
  return entry
# end def interpret_qmcpack_fname
