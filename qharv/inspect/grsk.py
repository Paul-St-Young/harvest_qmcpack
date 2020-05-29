import numpy as np

def ft_iso3d(myk, myr, frm):
  fkm = [np.trapz(myr*np.sin(k*myr)/k*frm, myr) for k in myk]
  return 4*np.pi*np.array(fkm)

def ift_iso3d(myr, myk, fkm):
  return ft_iso3d(myr, myk, fkm)/(2*np.pi)**3

def gr2sk(myk, myr, grm, rho):
  skm = 1+rho*ft_iso3d(myk, myr, grm-1)
  return skm

def sk2gr(myr, myk, skm, rho):
  grm = 1+ft_iso3d(myr, myk, skm-1)/rho/(2*np.pi)**3
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
  assert ndim == 3  # assume 3 dimensions
  # calculate volume of bins
  vnorm = np.diff(4*np.pi/3*bin_edges**ndim)
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
