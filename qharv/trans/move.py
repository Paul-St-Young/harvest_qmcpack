# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# move! pack up
#  useful routines for file transfer and file backup
import os
import subprocess as sp
from qharv.plantation.sugar import mkdir


def clean_path(path):
  """ remove any '.' or '..' in the path expression
  Args:
    path (str): any path string e.g. '../run/./opt'
  Return:
    str: clean path e.g. 'run/opt'
  """
  tokens = path.split('/')
  tokens1 = [tok for tok in tokens if tok not in ('.', '..')]
  path1 = '/'.join(tokens1)
  return path1


def source_target_map(flist, new_dir):
  """ for each file in flist, provide its new location in new_dir

  e.g.
    source_target_map(['./run/opt0', './run/opt1'], './opt_backup')

  Args:
    flist (list): a list of file locations
    new_dir (str): location of new directory
  Return:
    dict: st_map, source-target map for files operations such as copy or move.
  """
  st_map = {}
  for floc in flist:
    floc1 = os.path.join(new_dir, clean_path(floc))
    st_map[str(floc)] = str(floc1)
  return st_map


def inverse_st_map(flist, new_dir, old_dir):
  """ fix common screw up: moved all source files without saving st_map

  e.g.
    inverse_st_map(['./opt_backup/opt0', './opt_backup/opt1'],
                   './opt_backup',
                   './run')

  Args:
    flist (list): a list of backedup file locations
    new_dir (str): directory holding the backups
    old_dir (str): directory to restore backups to
  """
  parent_dir = os.path.dirname(old_dir)
  inv_st_map = source_target_map(flist, parent_dir)
  new_map = {}
  for floc, floc1 in inv_st_map.items():
    new_map[floc] = floc.replace(new_dir, parent_dir)
  return new_map


def move_by_st_map(st_map):
  """ move files according to source-target map

  Args:
    st_map (dict): source-target map
  """
  for floc, floc1 in st_map.items():
    path1 = os.path.dirname(floc1)
    mkdir(path1)
    sp.check_call(['mv', floc, floc1])
