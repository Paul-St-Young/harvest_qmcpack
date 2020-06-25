from qharv.reel import ascii_out

def test_key_value_pairs():
  text = '''
     the Fermi energy is     5.3030 ev

!    total energy              =      -2.13507393 Ry
     Harris-Foulkes estimate   =      -2.13507393 Ry
     estimated scf accuracy    <          6.6E-11 Ry

     The total energy is the sum of the following terms:

     one-electron contribution =       1.45702955 Ry
     hartree contribution      =       0.02977844 Ry
     xc contribution           =      -1.38179808 Ry
     ewald contribution        =      -2.23982313 Ry
     smearing contrib. (-TS)   =      -0.00026070 Ry
'''
  ref_dict = {
    '!    total energy': '-2.13507393 Ry',
    'Harris-Foulkes estimate': '-2.13507393 Ry',
    'one-electron contribution': '1.45702955 Ry',
    'hartree contribution': '0.02977844 Ry',
    'xc contribution': '-1.38179808 Ry',
    'ewald contribution': '-2.23982313 Ry',
    'smearing contrib. (-TS)': '-0.00026070 Ry'
  }
  ftxt = 'tmp.txt'
  with open(ftxt, 'w') as f:
    f.write(text)
  mm = ascii_out.read(ftxt)
  pairs = ascii_out.get_key_value_pairs(mm)
  mm.close()
  for key, val in ref_dict.items():
    assert pairs[key] == val
  for key, val in pairs.items():
    assert ref_dict[key] == val
