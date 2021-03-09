# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to process i-PI outputs

def interpret_headers(headers):
  columns = []
  ncol = 0
  for header in headers:
    toks = header.split()
    ict = toks[2]
    assert toks[3] == '-->'
    col = toks[4]
    # append column names
    if '-' in ict:
      i, j = list(map(int, ict.split('-')))
      for ic in range(j-i+1):
        columns.append('%s_%d' % (col, ic))
    else:
      assert int(ict) == ncol+1
      columns.append(col)
    # update number of columns
    ncol = len(columns)
  return columns

def parse_output(text, ncol_max=1000):
  import pandas as pd
  from qharv.reel.scalar_dat import get_string_io
  fp = get_string_io(text)
  # header lines describe columns
  headers = []
  for icol in range(ncol_max):
    header = fp.readline().strip()
    if header.startswith('# col'):
      headers.append(header)
    else:
      break
  # interpret column labels
  columns = interpret_headers(headers)
  df = pd.read_csv(fp, sep=r'\s+', header=None)
  fp.close()
  df.columns = columns
  return df
