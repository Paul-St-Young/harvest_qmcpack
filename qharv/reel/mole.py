# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# dig around for goodies
import os
import subprocess as sp
from lxml import etree

def files_with_regex(regex,rundir,case=True):
  """ find files with the given suffix in folder rundir
   rely on bash `find` command
  Args:
    regex (str):  regular expression for file names
    rundir (str): directory containing the files to be found
    case (bool, optional): case sensity, default is True
  Returns:
    list: flist, a list of filenames matching the given regular expression
  """
  popt = '-path'
  if not case:
    popt = '-ipath' # not case sensitive
  out = sp.check_output(['find',rundir,popt,regex])
  flist = out.split('\n')[:-1]
  return flist
# end def files_with_regex
def files_scalar_dat(calcdir):
  return files_with_regex('*scalar.dat',calcdir)
def files_stat_h5(calcdir):
  return files_with_regex('*stat.h5',calcdir)

def group_map(flist):
  """ group QMCPACK output files by series id
   if a file does not have a group index, then append it to an indepenednt list
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
