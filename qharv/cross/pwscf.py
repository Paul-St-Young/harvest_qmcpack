# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate QE pwscf results for use in QMCPACK
import os
import subprocess as sp
import numpy as np

def copy_charge_density(scf_dir, nscf_dir, execute=True):
  """Copy charge density files from scf folder to nscf folder.

  Args:
    scf_dir (str): scf folder
    nscf_dir (str): nscf folder
    execute (bool, optional): perform file system operations, default True
      if execute is False, then description of operations will be printed.
  """
  if scf_dir == nscf_dir:
    return  # do nothing
  from qharv.reel import mole
  from qharv.plantation.sugar import mkdir
  # find charge density
  fcharge = mole.find('*charge-density.dat', scf_dir)
  save_dir = os.path.dirname(fcharge)
  # find xml file with gvector description
  fxml = mole.find('*data-file*.xml', save_dir)  # QE 5 & 6 compatible
  save_rel = os.path.relpath(save_dir, scf_dir)
  save_new = os.path.join(nscf_dir, save_rel)
  # find pseudopotentials
  fpsps = mole.files_with_regex('*.upf', save_dir, case=False)
  if execute:  # start to modify filesystem
    mkdir(save_new)
    sp.check_call(['cp', fcharge, save_new])
    sp.check_call(['cp', fxml, save_new])
    for fpsp in fpsps:
      sp.check_call(['cp', fpsp, save_new])
  else:  # state what will be done
    path = os.path.dirname(fcharge)
    msg = 'will copy %s and %s' % (
      os.path.basename(fcharge), os.path.basename(fxml))
    if len(fpsps) > 0:
      for fpsp in fpsps:
        msg += ' and %s ' % fpsp
    msg += '\n to %s' % save_new
    print(msg)

def ktext_frac(kpts):
  """Write K_POINTS card assuming fractional kpoints with uniform weight.

  Args:
    kpts (np.array): kpoints in reciprocal lattice units
  Return:
    str: ktext to be fed into pw.x input
  """
  line_fmt = '%8.6f %8.6f %8.6f 1'
  nk = len(kpts)
  header = 'K_POINTS crystal\n%d\n' % nk
  lines = [line_fmt % (kpt[0], kpt[1], kpt[2]) for kpt in kpts]
  ktext = header + '\n'.join(lines)
  return ktext
