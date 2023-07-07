import numpy as np

def ft_iso(myk, r, fr, ndim=3):
  if ndim == 3:
    fk = [np.trapz(r*np.sin(k*r)/k*fr, r) for k in myk]
  elif ndim == 2:
    from scipy.special import j0
    fk = [np.trapz(j0(k*r)*r*fr, r) for k in myk]
  else:
    msg = 'ndim=%d' % ndim
    raise RuntimeError(msg)
  fk = np.array(fk)*2*(ndim-1)*np.pi
  return fk

def ift_iso(myr, k, fk, ndim=3):
  fr = ft_iso(myr, k, fk, ndim=ndim)
  fr /= (2*np.pi)**ndim
  return fr

def gr2sk(myk, myr, grm, rho, ndim=3):
  skm = 1+rho*ft_iso(myk, myr, grm-1, ndim=ndim)
  return skm

def sk2gr(myr, myk, skm, rho, ndim=3):
  grm = 1+ift_iso(myr, myk, skm-1, ndim=ndim)/rho
  return grm

def get_bin_edges(axes, rmin=0., rmax=None, nr=32):
  # create linear grid
  if rmax is None:
    from qharv.inspect.axes_pos import rwsc
    rmax = rwsc(axes)
  bin_edges = np.linspace(rmin, rmax, nr)
  return bin_edges

def get_gofr_norm(axes, bin_edges, n1, n2=None):
  from qharv.inspect.axes_pos import volume
  ndim, ndim = axes.shape
  # calculate volume of bins
  vnorm = np.diff(2*(ndim-1)/ndim*np.pi*bin_edges**ndim)
  # calculate density normalization
  if n2 is None:
    npair = n1*(n1-1)/2
  else:
    npair = n1*n2
  rho = npair/volume(axes)
  # assemble the norm vector
  gr_norm = 1./(rho*vnorm)
  return gr_norm

def ase_gofr(atoms, bin_edges, gr_norm):
  """Calculate the real-space pair correlation function g(r) among
   all pairs of atom types. Histogram distances along the radial
   direction, i.e. spherically averaged.

  Args:
    atoms (ase.Atoms): atoms
    bin_edges (np.array): histogram bin edges
    gr_norm (np.array): normalization of each bin
  Return:
    dict: gr1_map, one g(r) for each pair of atom types.
  Example:
    >>> axes = np.eye(3)
    >>> pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> atoms = Atoms('H2', cell=axes, positions=pos, pbc=1)
    >>> bin_edges = get_bin_edges(axes)
    >>> gr_norm = get_gofr_norm(axes, bin_edges, len(pos))
    >>> gr1_map = ase_gofr(atoms, bin_edges, gr_norm)
    >>> gr1 = gr1_map[(0, 0)]
    >>> r = 0.5*(bin_edges[1:]+bin_edges[:-1])
    >>> plt.plot(r, gr1)
    >>> plt.show()
  """
  from ase.geometry import get_distances
  ias = np.unique(atoms.get_atomic_numbers())
  gr1_map = {}  # snapshot g(r) between all pairs of particle types
  for i in range(len(ias)):
    for j in range(i, len(ias)):
      ia = ias[i]
      ja = ias[j]
      # select positions
      idx1 = [atom.index for atom in atoms if atom.number == ia]
      idx2 = [atom.index for atom in atoms if atom.number == ja]
      ni = len(idx1)
      nj = len(idx2)
      # calculate distances
      drij, rij = get_distances(
        atoms[idx1].get_positions(),
        p2=atoms[idx2].get_positions(),
        cell=atoms.get_cell(),
        pbc=1
      )
      # extract unique distances
      offset = 0
      if ia == ja:
        offset = 1
      idx = np.triu_indices_from(rij, offset)
      dists = rij[idx]
      # histogram
      hist, be = np.histogram(dists, bin_edges)
      gr1 = hist*gr_norm
      gr1_map[(ia, ja)] = gr1
  return gr1_map

def kshell_sels(kmags, zoom):
  """Select k-shells by magnitute

  Args:
    kmags (np.array): k-vector magnitutes
    zoom (float): zoom-in to resolve more shells, e.g.
      zoom=100 considers the second digit
  Return:
    list: a list of boolean masks, one for each shell.
  Example:
    >>> kshell_sels(np.array([1.231, 1.232, 1.233, 1.30]), 100)
    [[True, True, True, False], [False, False, False, True]]
    >>> kshell_sels(np.array([1.231, 1.232, 1.233, 1.30]), 1000)
    [[True, False, False, False],
     [False, True, False, False],
     [False, False, True, False],
     [False, False, False, True]]
  """
  kints = np.round(kmags*zoom).astype(int)
  unique_kints = np.unique(kints)
  nsh = len(unique_kints)
  sels = []
  for ish in range(nsh):
    kint = unique_kints[ish]  # shell integer label
    sel = kints == kint       # select this shell
    sels.append(sel)
  return sels

def shell_average(kvecs, ym, ye=None, zoom=100):
  kmags = np.linalg.norm(kvecs, axis=-1)
  sels = kshell_sels(kmags, zoom)
  nsh = len(sels)
  # loop over each shell and average
  uk = np.zeros(nsh)
  uym = np.zeros(nsh)
  if ye is not None:
    uye = np.zeros(nsh)
  for ish, sel in enumerate(sels):
    uk[ish] = np.mean(kmags[sel])
    uym[ish] = np.mean(ym[sel])
    if ye is not None:
      uye[ish] = np.sum(ye[sel]**2)**0.5/sel.sum()
  if ye is not None:
    return uk, uym, uye
  return uk, uym
