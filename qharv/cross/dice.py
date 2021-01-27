import pandas as pd
from qharv.reel import ascii_out

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
