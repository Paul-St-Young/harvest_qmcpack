# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to visualize volumetric data
import numpy as np
from qharv.field.kyrt import color_scatter

def figax3d(show_axis=True, label_axis=True, **kwargs):
  """ get a pair of fig and Axes3D
  similar to subplots() but for a single 3D figure

  Args:
    show_axis (bool, optional): show x, y, z axes and ticks, default True
  Return:
    tuple: matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot
  """
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d', **kwargs)
  if not show_axis:
    ax._axis3don = False
  if label_axis:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
  return fig, ax

def isosurf(ax, vol, level_frac=0.25):
    """ draw iso surface of volumetric data on matplotlib axis at given level

    Example usage:
      from mpl_toolkits.mplot3d import Axes3D # enable 3D projection
      vol = np.random.randn(10,10,10)
      fig = plt.figure()
      ax  = fig.add_subplot(1,1,1,projection='3d')
      isosurf(ax,vol)
      plt.show()

    Args:
      ax (plt.Axes3D): ax = fig.add_subplot(1,1,1,projection="3d")
      vol (np.array): 3D volumetric data having shape (nx,ny,nz)
      level_frac (float): 0.0->1.0, isosurface value as a fraction between min and max
    Returns:
      Poly3DCollection: mesh
    Effect:
      draw on ax """
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    nx, ny, nz = vol.shape
    lmin, lmax = vol.min(), vol.max()
    # select isosurface level
    level = lmin + level_frac*(lmax-lmin)
    if level < lmin or level > lmax:
        raise RuntimeError('level must be >%f and < %f' % (lmin, lmax))
    # make marching cubes
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        vol, level)
    # plot surface
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return mesh

def spline_volumetric(val3d):
  """ spline 3D volumetric data onto a unit cube

  Args:
    val3d (np.array): 3D volumetric data of shape (nx,ny,nz)
  Returns:
    RegularGridInterpolator: 3D function defined on the unit cube
  """
  from scipy.interpolate import RegularGridInterpolator
  nx, ny, nz = val3d.shape
  myx = np.linspace(0, 1, nx)
  myy = np.linspace(0, 1, ny)
  myz = np.linspace(0, 1, nz)
  fval3d = RegularGridInterpolator((myx, myy, myz), val3d)
  return fval3d

def frac_to_cart(frac, data):
  """ convert fractional to cartesian coordinates on regular grid
   useful for splined 3D data with RegularGridInterpolator

  Args:
    frac (np.array): fractional coordinates, shape (npt, ndim=3)
    data (dict): must contain 3D grid 'data', basis 'axes', and 'origin'
  Return:
    np.array: cartesian coordinates
  """
  nxyz = np.array(data['data'].shape)
  amat = (nxyz-1)*data['axes']
  rvec = np.dot(frac, amat)+data['origin']
  return rvec

def cart_to_frac(rvec, data):
  """ inverse of frac_to_cart """
  nxyz = np.array(data['data'].shape)
  amat = (nxyz-1)*data['axes']
  frac = np.dot(rvec-data['origin'], np.linalg.inv(amat))
  return frac

def axes_func_on_grid3d(axes, func, grid_shape):
  """ put a function define in axes units on a 3D grid
  Args:
    axes (np.array): dtype=float, shape=(3,3);
      3D lattice vectors in row major (i.e. a1 = axes[0])
    func (RegularGridInterpolator): 3D function defined on the unit cube
    grid_shape (np.array): dtype=int, shape=(3,); shape of real space grid
  Returns:
    grid (np.array): dtype=float, shape=grid_shape; volumetric data
  """
  from itertools import product  # iterate through grid points fast
  # make a cubic grid that contains the simulation cell
  grid = np.zeros(grid_shape)
  farthest_vec = axes.sum(axis=0)
  dxdydz = farthest_vec/grid_shape
  # fill cubic grid
  inv_axes = np.linalg.inv(axes)
  nx, ny, nz = grid_shape
  for i, j, k in product(range(nx), range(ny), range(nz)):
    rvec = np.array([i, j, k])*dxdydz
    uvec = np.dot(rvec, inv_axes)
    # skip points without data
    sel = (uvec > 1.) | (uvec < 0.)
    if len(uvec[sel]) > 0:
      continue
    grid[i, j, k] = func(uvec)
  return grid

def read_xsf_cell(fxsf, ndim=3):
  fp = open(fxsf, 'r')
  for line in fp:
    if 'PRIMVEC' in line:
      break
  data = []
  for l in range(ndim):
    line = fp.readline()
    data.append(line.split()[:ndim])
  axes = np.array(data, dtype=float)
  return axes

def read_xsf_datagrid_3d_density(
  fname,
  header='BEGIN_DATAGRID_3D',
  trailer='END_DATAGRID_3D'):
  """
  parse DATAGRID_3D block in xsf file

  Args:
    fname (str): xsf file name
    header (str): tag marking the beginning of grid
    trailer (str): tag marking the end of grid
  Return:
    np.array: data of 3D grid
  """
  from qharv.reel import ascii_out
  mm   = ascii_out.read(fname)
  text = ascii_out.block_text(mm, header, trailer)
  lines = text.split('\n')
  # first advance iline past metadata
  ncol_meta = 3
  iline = 0
  vecs = []
  for line in lines:
    tokens = line.split()
    if len(tokens) == ncol_meta:
      vecs.append(tokens)
    elif len(tokens) >= 4:  # density data
      break
    iline += 1
  # interpret metadata
  mesh= np.array(vecs[0], dtype=int)
  origin = np.array(vecs[1], dtype=float)
  axes = np.array(vecs[2:], dtype=float)
  # then convert data to density grid, which may be of unequal lengths
  all_numbers = [text.split() for text in lines[iline:-1]]
  # flatten before converting to np.array
  data = np.array([x for numbers in all_numbers for x in numbers], dtype=float)
  ret = dict(
    axes = axes,
    mesh = mesh,
    origin = origin,
    datagrid = data.reshape(mesh, order='F'),
  )
  return ret

def read_gaussian_cube(fcub):
  """
  Read Gaussian cube file

  example:
    entry = read_gaussian_cube('density.cub')
    data  = np.array(entry['data'])
    assert np.allclose(data.shape, entry['nxyz'])

  Args:
    fcub (str): cube file name
  Return:
    dict: dictionary of useful info
     [axes, elem, pos, nxyz, data]
  """
  nskip = 2  # skip 2 comment lines
  ndim = 3  # 3 spatial dimensions
  # hold entire file in memory
  with open(fcub, 'r') as f:
    text = f.read()
  # split into lines
  lines = text.split('\n')
  # read the number of atoms
  natom_line = lines[nskip]
  tokens = natom_line.split()
  natom = int(tokens[0])
  origin = np.array(tokens[1:], dtype=float)
  # read lattice vectors
  axes = []
  nxyz = []
  for idim in range(ndim):
    line   = lines[nskip+1+idim]
    tokens = line.split()
    nx = int(tokens[0])
    nxyz.append(nx)
    avec = np.array(tokens[-3:], dtype=float)
    axes.append(avec)
  # read atomic positions
  elem = []
  pos  = []
  for iatom in range(natom):
    line = lines[nskip+ndim+1+iatom]
    tokens = line.split()
    atom_number   = int(float(tokens[1]))
    atom_position = map(float, tokens[2:2+ndim])
    elem.append(atom_number)
    pos.append(atom_position)
  # density grid
  data = lines[nskip+ndim+natom+1:]
  data_text = ' '.join(data)
  data_vals = list(map(float, data_text.split()))

  nx, ny, nz = nxyz
  rgrid = np.array(data_vals, dtype=float).reshape(
    [nx, ny, nz], order='C')

  # turn file into dictionary
  entry = {'origin': origin, 'axes': axes,
           'elem': elem, 'pos': pos, 'data': rgrid}
  return entry

def write_gaussian_cube(fcub, data, overwrite=False, **kwargs):
  import os
  keys = data.keys()
  if os.path.isfile(fcub) and not overwrite:
    raise RuntimeError('%s exists' % fcub)
  # required inputs: grid axes (3, 3) matrix and volumetric data
  if 'axes' not in keys:
    raise RuntimeError('grid axes is required')
  if 'data' not in keys:
    raise RuntimeError('data grid is required')
  # optional inputs
  if 'elem' in keys:
    elem = data['elem']
  else:
    elem = (1,)
  if 'pos' in keys:
    pos = data['pos']
  else:
    pos = ((0, 0, 0),)
  if 'origin' in data.keys():
    origin = data['origin']
  else:
    origin = (0, 0, 0)
  text = write_gaussian_cube_text(
    data['data'], data['axes'],
    elem=elem, pos=pos, origin=origin, **kwargs)
  with open(fcub, 'w') as f:
    f.write(text)

def write_gaussian_cube_text(
  vol, axes,
  elem=(1,), qs=(0,), pos=((0, 0, 0),), origin=(0, 0, 0),
  two_line_comment='cube\nfile\n'):
  """Write Gaussian cube file using volumetric data

  Args:
    vol (np.array): volumetric data, shape (nx, ny, nz)
    axes (np.array): grid basis, e.g. np.diag((dx, dy, dz))
    elem (array-like, optional): list of atomic numbers, default (1,)
    qs (array-like, optional): list of atomic charges, default (0,)
    pos (array-like, optional): list of atomic positions
    origin (array-like, optional): coordinates of the origin
    two_line_comment (str, optional): comments at file head
  Return:
    str: Gaussian file content
  """
  text = two_line_comment
  # natom, origin
  natom = len(pos)
  x, y, z = origin
  line1 = '%4d %8.6f %8.6f %8.6f\n' % (natom, x, y, z)
  # grid, axes
  line2 = ''
  for n, vec in zip(vol.shape, axes):
    x, y, z = vec
    line2 += '%4d %8.6f %8.6f %8.6f\n' % (n, x, y, z)
  # atoms
  line3 = ''
  for num, q, vec in zip(elem, qs, pos):
    x, y, z = vec
    line3 += '%4d %4.1f %8.6f %8.6f %8.6f\n' % (num, q, x, y, z)
  # volumetric data (not human-readable format)
  #dline = ' '.join(vol.ravel().astype(str))
  dline = (len(vol.ravel())*'%8.6f ') % tuple(vol.ravel())
  return text + line1 + line2 + line3 + dline

def write_wavefront_obj(verts, faces, normals):
  """ save polygons in obj format

  obj format is more commonly used than ply

  Args:
    verts (np.array): shape=(nvert,3) dtype=float,
      vertices in cartesian coordinates.
    faces (np.array): shape=(nface,nside) dtype=int,
      polygons each specified as a list of vertices
      (in vertex coordinates defined by verts).
    normals (np.array): shape=(nvert,3) dtype=float,
      normal vectors used for smooth lighting.
      There is one normal vector per vertex.
  Returns:
    str: content of the obj file
  """
  text = ''
  if faces.dtype.kind != 'i':
    print('Warning: faces should be integers. Converting now.')
    faces = faces.astype(int)

  vert_fmt = '{name:s} {x:7.6f} {y:7.6f} {z:7.6f}\n'  # weights not supported
  # write vertices
  for ivert in range(len(verts)):
    vert  = verts[ivert]
    x, y, z = vert
    text += vert_fmt.format(name='v', x=x, y=y, z=z)
  # write normals
  for inorm in range(len(normals)):
    norm  = normals[inorm]
    x, y, z = norm
    text += vert_fmt.format(name='vn', x=x, y=y, z=z)
  # write faces
  face_fmt = '{name:s} {polyx:d}//{normalx:d} {polyy:d}//{normaly:d} {polyz:d}//{normalz:d}\n'  # texture not supported
  for iface in range(len(faces)):
    face  = faces[iface]
    x, y, z = face+1
    text += face_fmt.format(name='f', polyx=x, polyy=y, polyz=z,
      normalx=x, normaly=y, normalz=z)
  return text

def write_stanford_ply(verts, faces):
  """ save polygons in ply format

  ply is simpler than obj, but older and less used

  Args:
    verts (np.array): shape=(nvert,3) dtype=float,
      vertices in cartesian coordinates.
    faces (np.array): shape=(nface,nside) dtype=int,
      polygons each specified as a list of vertices
      (in vertex coordinates defined by verts).
  Returns:
    str: content of the ply file
  """
  from qharv.seed.xml import arr2text
  header = """ply
format ascii 1.0
element vertex {nvert:n}
property float x
property float y
property float z
element face {nface:d}
property list uchar int vertex_indices
end_header"""

  # !!!! assuming triangles in 3D
  ndim = 3
  nface = len(faces)
  new_faces = np.zeros([nface, ndim+1], dtype=int)
  new_faces[:, 0] = 3
  new_faces[:, 1:] = faces

  text = header.format(nvert=len(verts), nface=nface) + \
    arr2text(verts) + arr2text(new_faces).strip('\n')

  return text
