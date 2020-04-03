import numpy as np

def check_values(df):
  # compare parse data with reference
  ref   = np.arange(9).reshape(3, 3) + 1
  assert np.allclose(df.values, ref)

def check_columns(df):
  cols = df.columns.values
  expected_cols = ['x', 'y', 'z']
  for icol in range(3):
    assert cols[icol] == expected_cols[icol]

def test_no_header_read():
  # write test data to file
  fname = 'tmp.dat'
  fdat  = "1 2 3\n4 5 6\n7 8 9"
  with open(fname, 'w') as f:
    f.write(fdat)

  # parse ascii file
  from qharv.reel import scalar_dat
  df = scalar_dat.read(fname)
  check_values(df)

def test_header_read():
  # write test data to file
  fname = 'tmp.dat'
  fdat  = "# x y z\n1 2 3\n4 5 6\n7 8 9"
  with open(fname, 'w') as f:
    f.write(fdat)

  # parse ascii file
  from qharv.reel import scalar_dat
  df   = scalar_dat.read(fname)
  check_columns(df)
  check_values(df)

def test_read_to_list():
  # write test data to file
  fname = 'tmp.dat'
  fdat  = "# x y z\n1 2 3\n4 5 6\n7 8 9"
  fdat += "\n# x y z\n1 2 3\n4 5 6\n7 8 9"
  with open(fname, 'w') as f:
    f.write(fdat)

  # parse ascii file
  ref   = np.arange(9).reshape(3, 3) + 1
  from qharv.reel import scalar_dat
  dfl  = scalar_dat.read_to_list(fname)
  for df in dfl:
    check_columns(df)
    check_values(df)
