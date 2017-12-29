# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
import os
import subprocess as sp

def disperse(ginp_loc,calc_dir,execute=False,overwrite=False):
  """ disperse inputs bundled up in a grouped input
  Args:
    ginp_loc (str): location of grouped input e.g. ../runs/dmc/qmc.in
    calc_dir (str): folder to output new inputs e.g. dmc1
    execute (bool,optional): perform file I/O, default is False i.e. a dry run
    overwrite (bool,optional): overwrite existing files, default is False
  Returns:
    list: a list of new inputs
  """
  from qharv.seed import xml

  # path0 is the folder containing the current grouped input
  path0 = os.path.dirname(ginp_loc)
  calc_dir0 = os.path.basename(path0)
  # path  is the folder to contain the dispersed inputs
  path  = os.path.join( os.path.dirname(path0),calc_dir)
  if execute: # make folder if not there
    if not os.path.isdir(path): sp.check_call(['mkdir',path])
  # end if execute

  # for each input in grouped input file, add group text (gt) to project id
  #  if execute, write input in given folder
  flist = []
  with open(ginp_loc,'r') as f:
    ig = 0
    for line in f:
      # construct source and target input paths
      infile = line.strip('\n')
      floc0  = os.path.join(path0,infile)
      if not os.path.isfile(floc0):
        raise RuntimeError('%s not found'%floc0)
      floc   = os.path.join(path,infile)
      if os.path.isfile(floc) and (not overwrite) and execute:
        raise RuntimeError('%s exists; delete or overwrite '%floc)
      flist.append(floc)

      # modify prefix
      gt = 'g'+str(ig).zfill(3)
      doc = xml.read(floc0)
      pnode   = doc.find('.//project')
      prefix0 = pnode.get('id')
      prefix  = '.'.join([prefix0,gt])
      pnode.set('id',prefix)
      if execute: xml.write(floc,doc)

      ig += 1
    # end for line
  # end with open
  return flist
# end def disperse
