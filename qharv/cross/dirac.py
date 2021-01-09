import pandas as pd

# ====================== level 0: basic output ======================

def read(fout, vp_kwargs=None, mp_kwargs=None):
  from qharv.reel import ascii_out
  if vp_kwargs is None:
    vp_kwargs = dict()
  if mp_kwargs is None:
    mp_kwargs = dict()
  mm = ascii_out.read(fout)
  etot = ascii_out.name_sep_val(mm, 'Total energy', ':')
  ehomo = ascii_out.name_sep_val(mm, 'E(HOMO)', ':')
  elumo = ascii_out.name_sep_val(mm, 'E(LUMO)', ':')
  data = {
    'etot': etot,
    'ehomo': ehomo,
    'elumo': elumo,
  }
  if mm.find(b'* Vector print *') > 0:
    data['vectors'] = parse_vector_print(mm, **vp_kwargs)
  if mm.find(b'* Mulliken population analysis *') > 0:
    data['populations'] = parse_mulliken(mm, **mp_kwargs)
  mm.close()
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
  """Parse eigenvectors from DIRAC 'Vector print' .PRIVEC output

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

def parse_populations(mm, idxl):
  """Parse population from DIRAC 'Mulliken' .MULPOP output
   modified from parse_eigenvectors

  Args:
    mm (mmap.mmap): memory map of outputfile
    idxl (list): a list of starting memory locations for eigenvectors
  Return:
    pd.DataFrame: population information
  """
  from qharv.reel import ascii_out
  header = '--------------------------------------'
  trailer = 'Electronic eigenvalue no'
  entryl = []
  for i in idxl:
    mm.seek(i)
    line = mm.readline().decode()
    # eg. 'eigenvalue no.  2: -0.2364785578899  ('
    toks = line.split(':')
    left = toks[0]
    right = toks[1].split()[0]
    iev = int(left.split()[-1])
    ev = float(right)
    meta = {'iev': iev, 'ev': ev}
    # read body
    i0, i1 = ascii_out.locate_block(mm, header, trailer,
      force_tail=True, skip_trailer=True)
    if i1 < 0:
      i0, i1 = ascii_out.locate_block(mm, header, '**')
    # parse
    text = mm[i0:i1].decode()
    # e.g.
    # alpha    1.0000  |      1.0000
    # beta     0.0000  |      0.0000
    lines = text.split('\n')
    aline = lines[0]
    bline = lines[1]
    assert 'alpha' in aline
    assert 'beta' in bline
    atot = float(aline.split()[1])
    btot = float(bline.split()[1])
    entry = {'a_tot': atot, 'b_tot': btot}
    entry.update(meta)
    entryl.append(entry)
  df = pd.DataFrame(entryl)
  return df

def parse_mulliken(mm,
  header='* Mulliken population analysis *',
  mid_tag='Fermion ircop E1u',
  end_tag='** Total gross population **',
):
  from qharv.reel import ascii_out
  # seek to header
  idx = mm.find(header.encode())
  mm.seek(idx)
  # find all potential vectors to read
  idxl = ascii_out.all_lines_with_tag(mm, 'Electronic eigenvalue no.')
  # exclude extra lines
  iend = mm.find(end_tag.encode())
  if iend > 0:
    idxl = [i for i in idxl if i < iend]
  # partition into even and odd
  imid = mm.find(mid_tag.encode())
  idxg = [i for i in idxl if i < imid]
  idxu = [i for i in idxl if i >= imid]
  gdf = parse_populations(mm, idxg)
  gdf['mo_symm'] = 'E1g'
  udf = parse_populations(mm, idxu)
  udf['mo_symm'] = 'E1u'
  df = pd.concat([gdf, udf], sort=False).reset_index(drop=True)
  return df

def get_basis_symm(vdf):
  """Get basis symmetry labels

  Args:
    vdf (pd.DataFrame): Vector Print database
  Return:
    list: a list of strings indicating the symmetry of each basis function
  Example:
    >>> data = dirac.read('W6+_stu.out')
    >>> vdf = data['vectors']
    >>> ao_symms = dirac.get_basis_symm(vdf)
  """
  import numpy as np
  ibasl = vdf.ibas.unique()
  ibasl.sort()
  nbas = len(ibasl)
  assert np.allclose(np.array(ibasl)-1, range(nbas))
  ao_symms = []
  for ibas in ibasl:
    asymm = vdf.loc[vdf.ibas==ibas, 'ao_symm'].unique()
    assert len(asymm) == 1
    ao_symms.append(str(asymm[0]))
  return ao_symms

# =================== level 1: scalar relativistic ===================

def is_spinor_up(cup0, cdn0, ztol):
  """Determine if spinor is dominated by spin-up component

  Args:
    cup0 (np.array): complex valued spin-up component of spinor
    cdn0 (np.array): complex valued spin-down component of spinor
    ztol (float): small value to tolerate non-exact zero
  Return:
    bool: True if spinor is dominated by spin-up component
  """
  import numpy as np
  nup0 = np.dot(cup0, cup0.conj()).real
  ndn0 = np.dot(cdn0, cdn0.conj()).real
  is_up = True
  if nup0 < ztol:
    is_up = False
  elif ndn0 >= ztol:
    msg = '|cup| = %f; |cdn| = %f' % (nup0, ndn0)
    msg += ' neither is zero given ztol = %f.' % ztol
    raise RuntimeError(msg)
  return is_up

def read_scalar_relativistic(fdir, mu=None, ztol=1e-4, sort=True,
  renorm_dcoeff=False):
  """Read MO energy and coeff from DIRAC output
   assuming calculation is scalar relativistic

  Args:
    fdir (str): DIRAC output file
    mu (float, optional): chemical potential in eV, default 0.5*(HOMO+LUMO)
    ztol (float, optional): zero tolerance, default 1e-4
    sort (bool, optional): sort MOs by energy, default True
    renorm_dcoeff (bool, optional): renormalize d coefficiets for PySCF,
      default False
  Return:
    dict:
      etot (float): total energy
      ehomo (float): HOMO energy
      elumo (float): LUMO energy
      evals (np.array): MO energy
      cup (np.array): complex valued MO coeff for up orbitals
      cdn (np.array): complex valued MO coeff for down orbitals
  Example:
    >>> evals, cup, cdn = read_scalar_relativistic('W6+_stu.out', mu=np.inf)
  """
  import numpy as np
  data = read(fdir)
  if mu is None:  # return only occupied MOs by default
    mu = 0.5*(data['ehomo']+data['elumo'])
  df = data['vectors']
  nbas = df.ibas.max()
  sel0 = df.ev < mu
  el = []
  cupl = []
  cdnl = []
  for mo_symm in ['E1g', 'E1u']:
    sel1 = df.mo_symm == mo_symm
    ievl = df.loc[sel0&sel1, 'iev'].unique()
    for iev in ievl:
      sel = sel0&sel1&(df.iev==iev)
      evl = df.loc[sel, 'ev'].unique()
      assert len(evl) == 1
      ev = evl[0]
      el.append(ev)
      # 1-based to 0-based basis index
      idx = df.loc[sel, 'ibas'].values-1
      # up & dn components of this spinor
      cup0 = df.loc[sel, 'cup'].values
      cdn0 = df.loc[sel, 'cdn'].values
      # degenerate Kramer's pair
      cup1 = -cdn0.conj()
      cdn1 = cup0.conj()
      # is this spinor up or down?
      is_up = is_spinor_up(cup0, cdn0, ztol)
      if is_up:
        c = np.zeros(nbas, dtype=complex)
        c[idx] = cup0
        cupl.append(c)
        c = np.zeros(nbas, dtype=complex)
        c[idx] = cdn1
        cdnl.append(c)
      else:
        c = np.zeros(nbas, dtype=complex)
        c[idx] = cup1
        cupl.append(c)
        c = np.zeros(nbas, dtype=complex)
        c[idx] = cdn0
        cdnl.append(c)
  cup = np.array(cupl)
  cdn = np.array(cdnl)
  evals = np.array(el)
  if renorm_dcoeff:
    ao_symms = get_basis_symm(df)
    dnorm = 0.83775**0.5  # !!!! how to calculate this better?
    sel = [asymm.startswith('d') for asymm in ao_symms]
    cup[:, sel] /= dnorm
    cdn[:, sel] /= dnorm
  iup = np.ones(len(evals), dtype=bool)
  if sort:  # sort by MO energy
    iup = np.argsort(el)
  ret = {
    'etot': data['etot'],
    'ehomo': data['ehomo'],
    'elumo': data['elumo'],
    'evals': evals[iup],
    'cup': cup[iup].T,
    'cdn': cdn[iup].T,
  }
  return ret
