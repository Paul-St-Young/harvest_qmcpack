import pandas as pd

def read(fout, vp_kwargs=None):
  from qharv.reel import ascii_out
  if vp_kwargs is None:
    vp_kwargs = dict()
  mm = ascii_out.read(fout)
  etot = ascii_out.name_sep_val(mm, 'Total energy', ':')
  ehomo = ascii_out.name_sep_val(mm, 'E(HOMO)', ':')
  elumo = ascii_out.name_sep_val(mm, 'E(LUMO)', ':')
  data = {
    'etot': etot,
    'ehomo': ehomo,
    'elumo': elumo,
    'vectors': parse_vector_print(mm, **vp_kwargs)
  }
  return data

def parse_ev_text(text):
  lines = text.split('\n')
  entryl = []
  for line in lines:
    # eg. '1  L W   1 s      -1.1034620201  0.0000000000'
    toks = line.split()
    if len(toks) < 8:
      continue
    ibas = int(toks[0])
    elem = toks[2]
    symm = toks[4]
    # (a, b, c, d) -> a+ib, c+id
    cup = float(toks[-4])+1j*float(toks[-3])
    cdn = float(toks[-2])+1j*float(toks[-1])
    # Kramer's pair: (-c, d, a, -b)
    entry = {'elem': elem, 'ibas': ibas, 'ao_symm': symm,
             'cup': cup, 'cdn': cdn}
    entryl.append(entry)
  df = pd.DataFrame(entryl)
  return df

def parse_eigenvectors(mm, idxl):
  """Parse eigenvectors from DIRAC 'Vector print' output

  Args:
    mm (mmap.mmap): memory map of outputfile
    idxl (list): a list of starting memory locations for eigenvectors
  Return:
    pd.DataFrame: eigenvector information
  Example:
    >>> from qharv.reel import ascii_out
    >>> mm = ascii_out.read('inp_mol.out')
    >>> idx = mm.find(b'* Vector print *')
    >>> mm.seek(idx)
    >>> header = 'Electronic eigenvalue no.'
    >>> idxl = ascii_out.all_lines_with_tag(mm, header)
    >>> df = parse_eigenvectors(mm, idxl[:2])  # first two vectors
  """
  from qharv.reel import ascii_out
  header = '===================================================='
  trailer = 'Electronic eigenvalue no'
  dfl = []
  for i in idxl:
    mm.seek(i)
    line = mm.readline().decode()
    # eg. 'eigenvalue no.  2: -0.2364785578899'
    left, right = line.split(':')
    iev = int(left.split()[-1])
    ev = float(right)
    meta = {'iev': iev, 'ev': ev}
    # read body
    i0, i1 = ascii_out.locate_block(mm, header, trailer,
      force_tail=True, skip_trailer=True)
    if i1 < 0:
      i0, i1 = ascii_out.locate_block(mm, header, '*********')
    # parse
    text = mm[i0:i1].decode()
    df1 = parse_ev_text(text)
    for key, val in meta.items():
      df1[key] = val
    dfl.append(df1)
  df = pd.concat(dfl, axis=0).reset_index(drop=True)
  return df

def parse_vector_print(mm,
  header='* Vector print *',
  mid_tag='Fermion ircop E1u',
  end_tag='* Mulliken population analysis *',
):
  from qharv.reel import ascii_out
  # seek to header
  idx = mm.find(header.encode())
  mm.seek(idx)
  # find all potential vectors to read
  idxl = ascii_out.all_lines_with_tag(mm, 'Electronic eigenvalue no.')
  # exclude population analysis
  iend = mm.find(end_tag.encode())
  if iend > 0:
    idxl = [i for i in idxl if i < iend]
  # partition into even and odd
  imid = mm.find(mid_tag.encode())
  idxg = [i for i in idxl if i < imid]
  idxu = [i for i in idxl if i >= imid]
  gdf = parse_eigenvectors(mm, idxg)
  gdf['mo_symm'] = 'E1g'
  udf = parse_eigenvectors(mm, idxu)
  udf['mo_symm'] = 'E1u'
  df = pd.concat([gdf, udf], sort=False).reset_index(drop=True)
  return df
