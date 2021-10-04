import pandas as pd
from qharv.reel import ascii_out

# ====================== level 0: basic input =======================
def default_input(fdump,
  variational_kws=None, perturbation_kws=None, misc_kws=None):
  """ Example:
  >>> inp_text = default_input("FCIDUMP")
  """
  meta = read_fcidump_header(fdump)
  nmo = meta['NORB']
  nelec = meta['NELEC']
  ms2 = meta['MS2']
  nup = (nelec+ms2)//2
  ndn = nup-ms2
  assert nup+ndn == nelec

  text = '#system\nnocc %d\n' % nelec
  # default ROHF det
  iup = range(nup)
  idn = range(ndn)
  up_str = ' '.join(map(str, [2*i for i in iup]))
  dn_str = ' '.join(map(str, [2*i+1 for i in idn]))
  state_str = up_str + ' ' + dn_str
  text += state_str + "\nend\n"
  text += "orbitals %s\n\n" % fdump
  for kws, func in zip(
    [variational_kws, perturbation_kws, misc_kws],
    [variational_block, perturbation_block, misc_block]
  ):
    if kws is None:
      kws = dict()
    block = func(**kws)
    text += block
  return text

def read_fcidump_header(fdump, mline=1024):
  found_end = False
  # extract header text
  header = ''
  with open(fdump, 'r') as f:
    for line in f:
      if '=' in line:
        header += line
      if 'END' in line:
        found_end = True
        break
  if not found_end:
    msg = "FCIDUMP header longer than %d lines" % mline
    raise RuntimeError(msg)
  # parse header text
  lines = header.replace("&FCI", '').replace("&END", '').split("\n")
  meta = dict()
  for line in lines:
    tokens = line.split(',')
    for token in tokens:
      if "=" not in token:
        continue
      name, val = token.split("=")
      meta[name.strip()] = int(val)
  return meta

def variational_block(**kwargs):
  accepted_kws = {
    "davidsonTol": "%e",
    "dE": "%e",
    "maxIter": "%d",
    "nroots": "%d",
  }
  # set defaults
  schedule = kwargs.pop("schedule", None)
  if schedule is None:
    schedule = {0: 1e-4}
  maxIter = kwargs.pop("maxIter", None)
  if maxIter is None:
    kwargs["maxIter"] = max(schedule.keys())+1
  # connected space growth schedule
  text = "#variational\nschedule\n"
  for i, eps1 in schedule.items():
    line = "%d %e\n" % (i, eps1)
    text += line
  text += "end\n"
  # diagonalization settings
  for key, val in kwargs.items():
    if key not in accepted_kws:
      msg = 'unknown keyword "%s" in variational_block' % key
      raise RuntimeError(msg)
    fmt = accepted_kws[key]
    line = key + " " + fmt % val + "\n"
    text += line
  text += "\n"
  return text

def perturbation_block(**kwargs):
  accepted_kws = {
    "epsilon2": "%e",
    "targetError": "%e",
    "nPTiter": "%d",
    "sampleN": "%d",
  }
  nPTiter = kwargs.pop("nPTiter", None)
  if nPTiter is None:
    kwargs["nPTiter"] = 0
  sampleN = kwargs.pop("sampleN", None)
  if sampleN is None:
    kwargs["sampleN"] = 0
  text = '#pt\n'
  for key, val in kwargs.items():
    if key not in accepted_kws:
      msg = 'unknown keyword "%s" in perturbation_block' % key
      raise RuntimeError(msg)
    fmt = accepted_kws[key]
    line = key + " " + fmt % val + "\n"
    text += line
  text += "\n"
  return text

def misc_block(**kwargs):
  text = '#misc\n'
  for key, val in kwargs.items():
    line = "%s %s\n" % (key, str(val))
    text += line
  return text

# ====================== level 0: basic output ======================

def read_dice_output(fdat, force=False):
  mm = ascii_out.read(fdat)
  if not force:  # make sure run terminated normally
    idx = mm.find(b'Returning without error')
    if idx < 0:
      msg = 'Abnormal Dice termination %s' % fdat
      raise RuntimeError(fdat)
  data = {}
  eref = ascii_out.name_sep_val(mm, 'Ref. Energy', ':')
  data['eref'] = eref
  # read variational space
  df1 = parse_variational_step(mm)
  emin = df1.groupby(['Root', 'Eps1'])['Energy'].min()
  # get converged energy
  entryl = []
  for e in emin:
    sel = df1.Energy == e
    entryl.append(df1.loc[sel].iloc[-1])
  df2 = pd.DataFrame(entryl)
  for col in ['Iter', 'Root', 'ndet', 'ndave']:
    df2[col] = df2[col].astype(int)
  data['vdf'] = df2
  # look for perturbation
  idx = mm.find(b'Iter          EPTcurrent')
  if idx > 0:
    df3 = parse_perturbation_step(mm)
    emin = df3.groupby('State')['EPTcurrent'].min()
    entryl = []
    for e in emin:
      sel = df3['EPTcurrent'] == e
      entryl.append(df3.loc[sel].squeeze())
    df4 = pd.DataFrame(entryl)
    data['pdf'] = df4
  mm.close()
  return data

def read_rdm1(ftxt):
  import numpy as np
  with open(ftxt, 'r') as f:
    n = int(f.readline())
    dm = np.zeros([n, n])
    for line in f:
      it, jt, valt = line.split()
      i = int(it)
      j = int(jt)
      val = float(valt)
      dm[i, j] = val
  return dm

def parse_variational_step(mm):
  from qharv.reel import scalar_dat
  vtext = ascii_out.block_text(mm, 'Iter Root', 'Performing final tight',
    skip_header=False)
  ihead = vtext.find('\n')
  header = vtext[:ihead]
  body = vtext[ihead:]
  header = header.replace('#Var. Det.', 'ndet')
  header = header.replace('#Davidson', 'ndave')
  text = '# ' + header + '\n' + body
  df1 = scalar_dat.parse(text)
  return df1

def parse_perturbation_step(mm):
  from qharv.reel import scalar_dat
  ptext = ascii_out.block_text(mm, 'Iter          EPTcurrent  State',
    'Returning without error', skip_header=False, force_tail=True)
  lines = ptext.split('\n')
  header = lines[0]
  body = ''
  for line in lines[1:]:
    if len(line.split()) > 1:
      body += line + '\n'
    else:
      break
  text = '# ' + header + '\n' + body
  df1 = scalar_dat.parse(text)
  return df1
