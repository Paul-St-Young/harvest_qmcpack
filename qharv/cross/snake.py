# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to facilitate snakemake workflow.

def available_wildcards(pattern):
  """Find all wildcards that can be put into the pattern to locate
   existing files. Useful for finding finished runs in a folder without
   knowing the input parameters for each run.

  Args:
    pattern (str): snakemake path pattern
  Return:
    dict: map each wildcard name to a list of available values
  Example:
    >>> wcd = available_wildcards('runs/i{i}-j{j}/x{x}.dat')
    {'i': [1, 2], 'j': [3, 4], 'x': [0.1, 0.1]}
  """
  from snakemake.io import glob_wildcards
  wcs = glob_wildcards(pattern)
  wcd = wcs._asdict()
  return wcd

def hybrid_expand(regex, zips, **regs):
  """Combine expand zip with expand **kwargs. Allow correlated inputs (zips)
   and uncorrelated inputs (kwargs) to be set together.

  Args:
    regex (str): snakemake path pattern
    zips (dict): map each wildcard name to a list of the same length
    **kwargs: normal inputs to snakemake's expand
  Return:
    list: a list of paths
  Example:
    >>> hybrid_expand('i{i}-j{j}/x{x}.dat', {'i': [1, 2], 'j': [3, 4]}
    >>>  , x=[0.1, 0.2])
    ['i1-j3/x0.1.dat', 'i1-j3/x0.2.dat', 'i2-j4/x0.1.dat', 'i2-j4/x0.2.dat']
    >>> pattern = '{run_dir}/i{i}-j{j}/x{x}.dat'
    >>> wcd = available_wildcards(pattern)
    >>> hybrid_expand(pattern, wcd, run_dir='runs')
  """
  from snakemake.io import expand
  small = expand(regex, zip, **zips, allow_missing=True)
  big = expand(small, **regs)
  return big

def run_cmd(CMD, floc, osuf='.out', esuf='.err'):
  """
  Create bash command to run input file to output and error

  Args:
    CMD (str): execute command
    floc (str): input file location
  Return:
    str: bash command
  Example:
    >>> cmd = run_cmd('lih/scf.inp', 'mpirun -np 8 pw.x -nk 8 -in')
    >>> shell(cmd)
  """
  import os
  path = os.path.dirname(floc)
  finp = os.path.basename(floc)

  suf = finp[finp.rfind('.'):]
  cmd = 'cd %s; %s ' % (path, CMD)
  fout = finp.replace(suf, osuf)
  if fout == finp:
    msg = 'refuse to overwrite %s' % finp
    raise RuntimeError(msg)
  ferr = finp.replace(suf, esuf)
  cmd += '%s > %s 2> %s' % (finp, fout, ferr)
  return cmd
