# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Steal routines from atomic simulation environment (ase)
#  do NOT import this file within qharv, will add ase dependency!

from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from ase.visualize import view

from qharv.plantation import sugar


@sugar.check_file_before
def xsf2poscar(fpos, fxsf):
  s1 = read(fxsf, format='xsf')
  s1.write(fpos, format='vasp')
