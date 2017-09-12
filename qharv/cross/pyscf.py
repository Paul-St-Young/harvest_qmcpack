# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate pyscf results for use in QMCPACK

def atom_text(elem,pos):
  """  convert elem,pos to text representation

  for example, elem = ['C','C'], pos = [[0,0,0],[0.5,0.5,0.5]] will be 
  converted to 'C 0 0 0;\nC0.5 0.5 0.5'

  Args:
    elem (list): a list of atomic symbols such as 'H','C','O'
    pos  (list): a list of atomic positions, assume in 3D
  Returns:
    str: atomic string accepted by pyscf
  """
  assert len(elem) == len(pos)
  lines = []
  for iatom in range(len(elem)):
      mypos = pos[iatom]
      assert len(mypos) == 3
      line = '%5s  %10.6f  %10.6f  %10.6f' % (elem[iatom],mypos[0],mypos[1],mypos[2])
      lines.append(line)
  atext = ';\n'.join(lines)
  return atext
# end def
