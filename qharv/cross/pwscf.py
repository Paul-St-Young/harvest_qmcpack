# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate QE pwscf results for use in QMCPACK
import numpy as np

# ========================== level 0: read ==========================

def input_keywords(scf_in):
  """Extract all keywords from a quantum espresso input file

  Args:
    scf_in (str): path to input file
  Return:
    dict: a dictionary of inputs
  """
  keywords = dict()
  with open(scf_in, 'r') as f:
    for line in f:
      if '=' in line:
        key, val = line.split('=')
        keywords[key.strip()] = val.strip('\n')
  return keywords

# ========================= level 1: modify =========================

def change_keyword(text, section, key, val, indent=' '):
  """Change input keyword

  Args:
    text (str): input text
    section (str): section name, must be an existing section
     e.g. ['control', 'system', 'electrons', 'ions', 'cell']
    key (str): keyword name, e.g. ecutwfc, input_dft, nosym, noinv
    val (dtype): keyword value
  Return:
    str: modified input text
  """
  from qharv.reel import ascii_out
  # find section to edit
  sname = '&' + section
  if sname not in text:
    sname = '&' + section.upper()
  if sname not in text:
    msg = 'section %s not found in %s' % (section, text)
    raise RuntimeError(msg)
  # determine keyword data type
  if np.issubdtype(type(val), np.dtype(np.str).type):  # default to string
    fmt = '%s = "%s"'
  elif np.issubdtype(type(val), np.dtype(bool).type):
    fmt = '%s = %s'
    val = '.true.' if val else '.false.'
  elif np.issubdtype(type(val), np.integer):
    fmt = '%s = %d'
  elif np.issubdtype(type(val), np.floating):
    fmt = '%s = %f'
    if val < 1e-4:
      fmt = '%s = %e'
  else:
    msg = 'unknown datatype %s for "%s"' % (type(val), key)
    raise RuntimeError(msg)
  line = indent + fmt % (key, val)
  # edit input
  if key in text:  # change existing keyword
    text1 = ascii_out.change_line(text, key, line)
  else:  # put new keyword at beginning of section
    text1 = ascii_out.change_line(text, sname, sname+'\n'+line)
  return text1

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

# ====================== level 1: read output =======================

def get_converged_output(scf_out,
  conv_tag='End of self-consistent calculation',
  end_tag='init_run     :'
):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  idxl = ascii_out.all_lines_with_tag(mm, conv_tag)
  istart = idxl[-1]
  mm.seek(istart)
  iend = mm.find(end_tag.encode())
  return mm[istart:iend].decode()

def find_lines(lines, label):
  idxl = []
  for i, line in enumerate(lines):
    if label in line:
      idxl.append(i)
  return idxl

def parse_occupation_numbers(text):
  lines = text.split('\n')
  idxl = find_lines(lines, 'occupation numbers')
  omat = []
  for i in idxl:
    occl = []
    for line in lines[i+1:]:
      tokens = line.split()
      if len(tokens) < 1:
        break
      occl += list(map(float, tokens))
    omat.append(occl)
  return np.array(omat)

def read_occupation_numbers(scf_out):
  text = get_converged_output(scf_out)
  return parse_occupation_numbers(text)

def parse_bands(text):
  lines = text.split('\n')
  idxl = find_lines(lines, 'bands (ev)')
  bmat = []
  for i in idxl:
    evals = []
    for line in lines[i+2:]:
      tokens = line.split()
      if len(tokens) < 1:
        break
      evals += list(map(float, tokens))
    bmat.append(evals)
  return np.array(bmat)

def read_bands(scf_out):
  text = get_converged_output(scf_out)
  return parse_bands(text)

def parse_efermi(line):
  if 'the Fermi energy is' in line:
    efermi = float(line.split('is')[1].split()[0])
  elif 'Fermi energies are' in line:
    et = line.split('are')[1]
    eup, edn, ev = et.split()
    efermi = [float(eup), float(edn)]
  else:
    msg = 'unable to parse "%s"' % line
    raise RuntimeError(msg)
  return efermi

def read_efermi(scf_out, efermi_tag='the Fermi energ'):
  text = get_converged_output(scf_out)
  for line in text.split('\n'):
    if efermi_tag in line:
      break
  return parse_efermi(line)

def parse_kline(line, ik=None):
  from qharv.reel import ascii_out
  assert 'k(' in line
  ikt, kvect, wkt = line.split('=')
  myik = int(ascii_out.lr_mark(ikt, '(', ')'))
  if ik is not None:  # check k index
    assert ik == myik-1  # fortran 1-based indexing
  wk = float(wkt)
  klist = ascii_out.lr_mark(kvect, '(', ')').split()
  kvec = np.array(klist, dtype=float)
  return kvec, wk

def read_kpoints(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  # get lattice units
  alat = ascii_out.name_sep_val(mm, 'lattice parameter (alat)')
  blat = 2*np.pi/alat
  # start parsing k points
  idx = mm.find(b'number of k points')
  mm.seek(idx)
  # read first line
  #  e.g. number of k points=    32  Fermi-Dirac smearing ...
  line = mm.readline().decode()
  nk = int(line.split('=')[1].split()[0])
  # confirm units in second line
  line = mm.readline().decode()
  assert '2pi/alat' in line
  # start parsing kvectors
  data = np.zeros([nk, 4])  # ik, kx, ky, kz, wk
  for ik in range(nk):
    line = mm.readline().decode()
    kvec, wk = parse_kline(line, ik=ik)
    data[ik, :3] = kvec*blat
    data[ik, 3] = wk
  mm.close()
  return data

def read_polar_mag(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  natom = ascii_out.name_sep_val(mm, 'number of atoms/cell', dtype=int)
  idx = ascii_out.all_lines_with_tag(mm, 'polar coord.:')
  lines = ascii_out.all_lines_at_idx(mm, idx)
  mm.close()
  data = []
  for line in lines:
    tokens = line.split(':')[-1].split()
    data.append(list(map(float, tokens)))
  polars = np.array(data).reshape(-1, natom, 3)
  return polars

def read_mag_per_site(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  natom = ascii_out.name_sep_val(mm, 'number of atoms/cell', dtype=int)
  idx = ascii_out.all_lines_with_tag(mm, "Magnetic moment per site")
  mags = np.empty([len(idx), natom])
  for iscf, i in enumerate(idx):
    mm.seek(i)
    mm.readline()
    for iatom in range(natom):
      line = mm.readline().decode()
      ct = line.split("charge=")[1].split()[0]
      mt = line.split("magn=")[1].split()[0]
      chg = float(ct)
      mag = float(mt)
      mags[iscf, iatom] = mag/chg
  return mags

# ========================= level 2: cross ==========================

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
  import os
  import subprocess as sp
  from qharv.reel import mole
  from qharv.field.sugar import mkdir
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
