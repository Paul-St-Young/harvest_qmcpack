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
    text = f.read()
  return parse_keywords(text)

def parse_keywords(text):
  keywords = dict()
  for line in text.split('\n'):
    if '=' in line:
      key, val = line.split('=')
      val1 = val.strip('\n').strip()
      val2 = val1.strip("'").strip('"')
      keywords[key.strip()] = val2
  return keywords

# ========================= level 1: modify =========================

def change_keyword(text, section, key, val, indent=' ', float_fmt='%e'):
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
    fmt = '%s = ' + float_fmt
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
  line_fmt = '%16.10f %16.10f %16.10f 1'
  nk = len(kpts)
  header = 'K_POINTS crystal\n%d\n' % nk
  lines = [line_fmt % (kpt[0], kpt[1], kpt[2]) for kpt in kpts]
  ktext = header + '\n'.join(lines)
  return ktext

def cell_parameters(axes, unit='bohr', fmt='%24.16f'):
  cell_text = '\nCELL_PARAMETERS %s\n' % unit
  ndim = len(axes)
  for a in axes:
    line = (fmt*ndim + '\n') % tuple(a)
    cell_text += line
  return cell_text

# ===================== level 1: file locations =====================

def get_prefix_outdir(scf_in):
  inps = input_keywords(scf_in)
  prefix = inps.pop("prefix", "pwscf")
  outdir = inps.pop("outdir", ".")
  return prefix, outdir

def find_xml(scf_inp):
  import os
  prefix, outdir = get_prefix_outdir(scf_inp)
  path = os.path.dirname(scf_inp)
  fxml = os.path.join(path, outdir, prefix) + ".xml"
  if not os.path.isfile(fxml):
    msg = "%s not found" % fxml
    raise RuntimeError(msg)
  return fxml

def find_save(scf_inp):
  import os
  prefix, outdir = get_prefix_outdir(scf_inp)
  path = os.path.dirname(scf_inp)
  dsave = os.path.join(path, outdir, prefix) + ".save"
  if not os.path.isdir(dsave):
    msg = "%s not found" % dsave
    raise RuntimeError(msg)
  return dsave

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

def polar2cart(polar):
  """Example:
  >>> polars = read_polar_mag(scf_out)
  >>> vels = [polar2cart(pol) for pol in polars]
  """
  r, theta, phi = polar.T
  theta *= np.pi/180
  phi *= np.pi/180
  z = r*np.cos(theta)
  xy = r*np.sin(theta)
  x = xy*np.cos(phi)
  y = xy*np.sin(phi)
  xyz = np.c_[x, y, z]
  return xyz

def cart2polar(cart):
  """Inverse of polar2cart
  """
  x, y, z = cart.T
  r = np.linalg.norm(cart, axis=-1)
  theta = np.arccos(z/r)
  phi = np.arctan2(y, x)
  # convert to degree
  theta = theta/np.pi*180
  phi = phi/np.pi*180
  pol = np.c_[r, theta, phi]
  return pol

def read_polar_chg(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  natom = ascii_out.name_sep_val(mm, 'number of atoms/cell', dtype=int)
  idx = ascii_out.all_lines_with_tag(mm, 'relative position')
  data = []
  for i in idx:
    mm.seek(i)
    mm.readline()
    line = mm.readline().decode()
    assert 'charge' in line
    valt = line.split(':')[1].split()[0]
    val = float(valt)
    data.append(val)
  chgs = np.array(data).reshape(-1, natom)
  return chgs

def read_chgmag_per_site(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  natom = ascii_out.name_sep_val(mm, 'number of atoms/cell', dtype=int)
  idx = ascii_out.all_lines_with_tag(mm, "Magnetic moment per site")
  chgs = np.empty([len(idx), natom])
  mags = np.empty([len(idx), natom])
  for iscf, i in enumerate(idx):
    mm.seek(i)
    mm.readline()
    for iatom in range(natom):
      line = mm.readline().decode()
      ct = line.split("charge=")[1].split()[0]
      mt = line.split("magn=")[1].split()[0]
      chg = float(ct)
      chgs[iscf, iatom] = chg
      mag = float(mt)
      mags[iscf, iatom] = mag
  return chgs, mags

# ========================= level 2: cross ==========================
def link_save(scf_inp, path):
  """Link {prefix}.save folder from pwscf run to new folder

  Args:
    scf_inp (str): pwscf input file
    path (str): path to new folder
  Example:
    >>> link_save('scf.inp', '../convert/p2q')
  """
  import os
  import subprocess as sp
  # find wf save
  prefix, outdir = get_prefix_outdir(scf_inp)
  dsave = find_save(scf_inp)
  # link to converter save location
  outpath = os.path.join(path, outdir)
  hsave = os.path.join(outpath, '%s.save' % prefix)
  if not (os.path.abspath(hsave) == dsave):
    rpath = os.path.relpath(dsave, outpath)
    if os.path.isdir(outpath):
      msg = '%s exists' % outpath
      raise RuntimeError(msg)
    sp.check_call(['mkdir', outpath])
    cmd = 'cd %s; ln -s %s %s.save' % (outpath, rpath, prefix)
    sp.check_call(cmd, shell=True)

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
  try:
    fcharge = mole.find('*charge-density.hdf5', scf_dir)
  except RuntimeError:
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
