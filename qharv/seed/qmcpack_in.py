# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
import os
import subprocess as sp
from lxml import etree

from qharv.reel import mole
from qharv.seed import xml, xml_examples

# =============== level 0: build input from scratch ===============
def assemble_project(nodel, name='qmc', series=0):
  """ assemble QMCPACK input using a list of xml nodes

  usually nodel=[qmcsystem, qmc]

  Args:
    nodel (list): a list of xml node (lxml.Element)
    name (str, optional): project name, default 'qmc'
  """
  qsim = etree.Element('simulation')
  proj = xml.make_node('project', {'id': name, 'series': str(series)})
  for node in [proj]+nodel:
    qsim.append(node)
  doc = etree.ElementTree(qsim)
  return doc

def simulationcell_from_axes(axes, bconds='p p p', rckc=15.):
  """ construct the <simulationcell> xml element from axes

   Args:
     axes (np.array): lattice vectors
     bconds (str, optional): boundary conditions in x,y,z directions.
      p for periodic, n for non-periodic, default to 'p p p'
     rckc: long-range cutoff paramter rc*kc, default to 15
   Return:
     etree.Element: representing <simulationcell>
  """

  def pad_line(line):  # allow content to be selected by double clicked
    return ' ' + line + ' '

  # write primitive lattice vectors
  lat_node = etree.Element('parameter', attrib={
    'name': 'lattice',
    'units': 'bohr'
  })
  lat_node.text = xml.arr2text(axes)

  # write boundary conditions
  bconds_node = etree.Element('parameter', {'name': 'bconds'})
  bconds_node.text = pad_line(bconds)

  # write long-range cutoff parameter
  lr_node = etree.Element('parameter', {'name': 'LR_dim_cutoff'})
  lr_node.text = pad_line(str(rckc))

  # build <simulationcell>
  sc_node = etree.Element('simulationcell')
  sc_node.append(lat_node)
  sc_node.append(bconds_node)
  sc_node.append(lr_node)
  return sc_node

def pos_attrib(pos):
  """ consturct <attrib name="positions">

  Args:
    pos (np.array): positions, shape (nptcl, ndim)
  Return:
     etree.Element: <attrib name="positions">
  """
  pa = xml.etree.Element('attrib', dict(
    name = 'position',
    datatype = 'posArray',
    condition = str(0)
  ))
  pa.text = xml.arr2text(pos)
  return pa

def particle_group_from_pos(pos, name, charge, **kwargs):
  """ construct a <group> in the <particleset> xml element

   Args:
     pos (np.array): positions, shape (nptcl, ndim)
     name (str): name of particle group
     charge (float): the amount of charge of this particle species
   Return:
     etree.Element: <group> including <attrib name="positions">
  """
  if 'charge' in kwargs:
    msg = 'keyword charge %f will be overwriten'
    msg += ' by %f' % (kwargs['charge'], charge)
    raise RuntimeError(msg)
  kwargs['charge'] = charge
  group = xml.etree.Element('group', dict(
    name = name,
    size = str(len(pos)),
  ))
  for key, val in kwargs.items():
    xml.set_param(group, key, str(val), new=True)
  pa = pos_attrib(pos)
  group.append(pa)
  return group

def ud_electrons(nup, ndown):
  """ construct the <particleset name="e"> xml element for electrons

   Args:
     nup (int): number of up electrons
     ndown (int): number of down electrons
   Return:
     etree.Element: representing <particleset name="e">
  """

  epset = etree.Element('particleset', {'name': 'e', 'random': 'yes'})

  up_group = etree.Element('group', {
    'name': 'u',
    'size': str(nup),
    'mass': '1.0'
  })
  dn_group = etree.Element('group', {
    'name': 'd',
    'size': str(ndown),
    'mass': '1.0'
  })
  for egroup in [up_group, dn_group]:
    xml.set_param(egroup, 'charge', ' -1 ', new=True)
    epset.append(egroup)

  return epset

def all_electron_hamiltonian(elec_name='e', ion_name='ion0'):
  ee = xml.make_node('pairpot', {
    'type': 'coulomb',  'name': 'ElecElec',
    'source': elec_name, 'target': elec_name
  })
  ham = xml.make_node('hamiltonian')
  ham.append(ee)
  if ion_name is not None:
    ei = xml.make_node('pairpot', {
      'type': 'coulomb',  'name': 'ElecIon',
      'source': ion_name, 'target': elec_name
    })
    ii = xml.make_node('pairpot', {
      'type': 'coulomb',  'name': 'IonIon',
      'source': ion_name, 'target': ion_name
    })
    xml.append(ham, [ei, ii])
  return ham

def bspline_qmcsystem(fh5, tmat=None):
  """Create Slater-Jastrow system input from pw2qmcpack.x h5 file

  Args:
    fh5 (str): path to wf h5 file
    tmat (np.array): tile matrix
  Return:
    qsys (etree.Element): <qmcsystem>
  """
  import numpy as np
  from qharv.seed import wf_h5
  ndim0 = 3  # !!!! hard-code for three dimensions
  fp = wf_h5.read(fh5)
  axes, elem, charge_map, pos = wf_h5.axes_elem_charges_pos(fp)
  nelecs = wf_h5.get(fp, 'nelecs')
  fp.close()
  natom, ndim = pos.shape
  assert ndim == ndim0
  nup, ndn = nelecs
  if nup != ndn:  # hard-code for unpolarized for now
    raise RuntimeError('nup != ndn')
  if tmat is None:  # use primitive cell by default
    tmat = np.eye(3, dtype=int)
  else:  # tile supercell
    from qharv.inspect.axes_elem_pos import ase_tile
    axes, elem, pos = ase_tile(axes, elem, pos, tmat)
  spoup = spodn = 'spo_ud'
  psi_name = 'psi0'
  ion_name = 'ion0'
  nodes = []

  # simulationcell
  sc_node = simulationcell_from_axes(axes)
  nodes.append(sc_node)

  # particlesets
  ions = xml.make_node('particleset', {'name': ion_name})
  for name, charge in charge_map.items():
    sel = elem == name
    ion_grp = particle_group_from_pos(pos[sel], name, charge)
    ions.append(ion_grp)
  nodes.append(ions)
  elecs = ud_electrons(*nelecs)
  nodes.append(elecs)

  # sposet and builder
  tmat_str = ('%d ' * 9) % tuple(tmat.ravel())
  sb = xml.make_node('sposet_builder', {
      'type': 'bspline',
      'href': fh5,
      'tilematrix': tmat_str,
      'twistnum': '0',
      'source': ion_name,  # "Einspline needs the source particleset"
  })
  spo = xml.make_node('sposet', {
    'type': 'bspline',
    'name': spoup,
    'size': str(nup),
    'spindataset': '0'}
  )
  sb.append(spo)

  # determinantset
  updet = xml.make_node('determinant', {
    'id': 'updet',
    'size': str(nup),
    'sposet': spoup
  })
  dndet = xml.make_node('determinant', {
    'id': 'dndet',
    'size': str(ndn),
    'sposet': spodn
  })
  sdet = xml.make_node('slaterdeterminant')
  xml.append(sdet, [updet, dndet])
  dset = xml.make_node('determinantset')
  dset.append(sdet)

  # wave function
  wf = xml.make_node('wavefunction', {'name': psi_name, 'target': 'e'})
  xml.append(wf, [sb, dset])
  nodes.append(wf)

  # hailtonian
  ham = all_electron_hamiltonian()
  nodes.append(ham)
  qsys = xml.make_node('qmcsystem')
  xml.append(qsys, nodes)
  return qsys

# ================== level 1: use existing input ===================
def expand_twists(example_in_xml, twist_list, calc_dir, force=False):
  """ expand example input xml to all twists in twist_list

  Naming convention of new inputs:
    [prefix].g001.twistnum_[itwist].in.xml

  Args:
    example_in_xml (str): example QMCPACK input xml file
    twist_list (list): a list of twist indices
    calc_dir (str): folder to output new inputs
  Return:
    None
  Examples:
    >>> expand_twists('./vmc.in.xml',range(64),'.')
    >>> expand_twists('./ref/vmc.in.xml',[0,15,17],'./new')
  """
  doc    = xml.read(example_in_xml)
  prefix = doc.find('.//project').get('id')

  fname_fmt = '{prefix:s}.{gt:s}.twistnum_{itwist:d}.in.xml'
  bundle_text = ''
  bundle_in = os.path.join(calc_dir, '%s.in' % prefix)
  for itwist in twist_list:
    # change twist number
    bb = doc.find('.//sposet_builder[@type="bspline"]')
    if bb is None:  # assume old-style input
      bb = doc.find('.//determinantset[@type="bspline"]')
    bb.set('twistnum', str(itwist))

    # construct file name
    gt = 'g' + str(itwist).zfill(3)
    fname = fname_fmt.format(
      prefix = prefix,
      gt     = gt,
      itwist = itwist
    )
    floc = os.path.join(calc_dir, fname)
    bundle_text += '%s\n' % fname

    if not force:  # check if file exists
      if os.path.isfile(floc):
        raise RuntimeError('force to overwrite %s' % floc)
    xml.write(floc, doc)
  with open(bundle_in, 'w') as f:
    f.write(bundle_text)

def bundle_twists(calc_dir, fregex='*twistnum_*.in.xml'):
  """ bundle all twist inputs

  quick and dirty: `cd $calc_dir; ls > prefix.in`, then edit prefix.in

  Args:
    calc_dir (str): calculation directory
    fregex (str, optional): regular expression for all twists
  Return:
    str: bundled input text
  """
  flist = mole.files_with_regex(fregex, calc_dir, maxdepth=1)
  flist.sort()

  text = ''
  for floc in flist:
    fname = os.path.basename(floc)
    text += fname + '\n'
  return text


def disperse(ginp_loc, calc_dir, execute=False, overwrite=False):
  """ disperse inputs bundled up in a grouped input

  Args:
    ginp_loc (str): location of grouped input e.g. ../runs/dmc/qmc.in
    calc_dir (str): folder to output new inputs e.g. dmc1
    execute (bool,optional): perform file I/O, default is False i.e. a dry run
    overwrite (bool,optional): overwrite existing files, default is False
  Returns:
    list: a list of new inputs
  """

  # path0 is the folder containing the current grouped input
  path0 = os.path.dirname(ginp_loc)
  calc_dir0 = os.path.basename(path0)
  # path  is the folder to contain the dispersed inputs
  path  = os.path.join(os.path.dirname(path0), calc_dir)
  if execute:  # make folder if not there
    if not os.path.isdir(path):
      sp.check_call(['mkdir', path])

  # for each input in grouped input file, add group text (gt) to project id
  #  if execute, write input in given folder
  flist = []
  with open(ginp_loc, 'r') as f:
    ig = 0
    for line in f:
      # construct source and target input paths
      infile = line.strip('\n')
      floc0  = os.path.join(path0, infile)
      if not os.path.isfile(floc0):
        raise RuntimeError('%s not found' % floc0)
      floc   = os.path.join(path, infile)
      if os.path.isfile(floc) and (not overwrite) and execute:
        raise RuntimeError('%s exists; delete or overwrite ' % floc)
      flist.append(floc)

      # modify prefix
      gt = 'g'+str(ig).zfill(3)
      doc = xml.read(floc0)
      pnode   = doc.find('.//project')
      prefix0 = pnode.get('id')
      prefix  = '.'.join([prefix0, gt])
      pnode.set('id', prefix)
      if execute:
        xml.write(floc, doc)

      ig += 1
  return flist

def set_norb(doc, norb):
  """ occupy the lowest norb Kohn-Sham orbitals

  Args:
    doc (etree.Element): must contain <particleset>, <sposet>, and <det...set>
    norb (int): number of orbitals to occupy
  Return:
    None
  Effect:
    doc is modified
  """
  epset = doc.find('.//particleset[@name="e"]')
  for group in epset.findall('.//group'):  # 'u' and 'd'
    group.set('size', str(norb))

  sposet = doc.find('.//sposet[@name="spo_ud"]')
  sposet.set('size', str(norb))

  detset = doc.find('.//determinantset')
  for det in detset.findall('.//determinant'):
    det.set('size', str(norb))

def set_gc_occ(norbl, calc_dir, fregex_fmt='*twistnum_{itwist:d}.in.xml'):
  """ edit twist inputs in calc_dir according to occupation vector norbl

  Args:
    norbl (list): number of occupied orbitals at each twist
    calc_dir (str): location of twist inputs
  """
  nocc = len(norbl)
  wild_fregex = fregex_fmt.replace('{itwist:d}', '*')
  flist = mole.files_with_regex(wild_fregex, calc_dir)
  ntwist = len(flist)
  if nocc != ntwist:
    raise RuntimeError('%d occupations given for %d twists' % (nocc, ntwist))

  for itwist in range(ntwist):
    norb = int(norbl[itwist])
    fregex = fregex_fmt.format(itwist=itwist)
    fxml = mole.find(fregex, calc_dir)
    doc = xml.read(fxml)
    set_norb(doc, norb)
    xml.write(fxml, doc)

def set_nwalker(doc, nwalker):
  """ set the number of walkers to use in DMC

  Args:
    doc (lxml.Element): xml node containing <qmc>
    nwalker (int): number of walkers
  """
  nodes = doc.findall('.//qmc[@method="vmc"]')
  for node in nodes:
    xml.set_param(node, 'samples', str(nwalker))
  nodes = doc.findall('.//qmc[@method="dmc"]')
  for node in nodes:
    xml.set_param(node, 'targetwalkers', str(nwalker))
