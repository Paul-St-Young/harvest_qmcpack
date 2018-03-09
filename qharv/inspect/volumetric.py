# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to visualize volumetric data
import numpy as np

def isosurf(ax,vol,level_frac=0.25):
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
    nx,ny,nz = vol.shape
    lmin,lmax = vol.min(),vol.max()

    level = lmin + level_frac*(lmax-lmin)
    if level<lmin or level>lmax:
        raise RuntimeError('level must be >%f and < %f'%(lmin,lmax))
    # end if

    # make marching cubes
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        vol, level)

    # plot surface
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.set_zlim(0,nz)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return mesh
# end def isosurf

def color_scatter(ax,data,cmap_name='viridis',**kwargs):
  """ view sampled 3D scalar function using value as color
  Args:
    ax (plt.Axes3D): ax = fig.add_subplot(1,1,1,projection="3d")
    data (np.array): aligned scatter data, the columns are [x,y,z,f(x,y,z)]
    cmap_name (str, optional): color map name, default is 'viridis'
    kwargs (dict, optional): keyword arguments to be passed to ax.scatter
  Returns:
    mpl_toolkits.mplot3d.art3d.Path3DCollection: scatter plot
  """
  x,y,z,val = data.T  # aligned scatter data

  # design color scheme, if none given
  if (not 'c' in kwargs.keys()) and (not 'color' in kwargs.keys()):
    from qharv.plantation import kyrt
    v2c = kyrt.scalar_color_map(min(val),max(val),cmap_name)
    kwargs['c'] = v2c(val)
  # end if

  # scatter
  s = ax.scatter(x,y,z,val,**kwargs)
  return s
# end def color_scatter

def spline_volumetric(val3d):
  """ spline 3D volumetric data onto a unit cube

  Args:
    val3d (np.array): 3D volumetric data of shape (nx,ny,nz)
  Returns:
    RegularGridInterpolator: 3D function defined on the unit cube
  """
  from scipy.interpolate import RegularGridInterpolator
  nx,ny,nz = val3d.shape
  myx = np.linspace(0,1,nx)
  myy = np.linspace(0,1,ny)
  myz = np.linspace(0,1,nz)
  fval3d = RegularGridInterpolator((myx,myy,myz),val3d)
  return fval3d
# end def spline_volumetric

def axes_func_on_grid3d(axes,func,grid_shape):
  """ put a function define in axes units on a 3D grid
  Args:
    axes (np.array): dtype=float, shape=(3,3); 3D lattice vectors in row major (i.e. a1 = axes[0])
    func (RegularGridInterpolator): 3D function defined on the unit cube
    grid_shape (np.array): dtype=int, shape=(3,); shape of real space grid
  Returns:
    grid (np.array): dtype=float, shape=grid_shape; volumetric data
  """
  from itertools import product # iterate through grid points fast

  # make a cubic grid that contains the simulation cell
  grid = np.zeros(grid_shape)
  farthest_vec = axes.sum(axis=0)
  dxdydz = farthest_vec/grid_shape

  # fill cubic grid
  inv_axes = np.linalg.inv(axes)
  nx,ny,nz = grid_shape
  for i,j,k in product(range(nx),range(ny),range(nz)):
    rvec = np.array([i,j,k])*dxdydz
    uvec = np.dot(rvec,inv_axes)

    # skip points without data
    sel = (uvec>1.) | (uvec<0.)
    if len(uvec[sel])>0:
      continue
    # end if

    grid[i,j,k] = func(uvec)
  # end for i,j,k
  return grid
# end def axes_func_on_grid3d

def xsf_datagrid_3d_density(fname,header='BEGIN_DATAGRID_3D_density',trailer='END_DATAGRID_3D_density'):
  from qharv.reel import ascii_out
  mm   = ascii_out.read(fname)
  text = ascii_out.block_text(mm,header,trailer)

  lines = text.split('\n')

  # first advance iline past particle coordinates (!!!! hacky)
  iline = 0
  for line in lines:
    if iline == 0:
      grid_shape = map(int,lines[0].split())
      iline += 1
      continue
    # end if

    tokens = line.split()
    if len(tokens) == 3: # atom coordinate
      pass
    elif len(tokens) >= 4: # density data
      break
    # end if
    iline += 1
  # end for line

  # then convert data to density grid, which may be of unequal lengths
  all_numbers = [text.split() for text in lines[iline:-1]]

  # flatten before converting to np.array
  data = np.array([x for numbers in all_numbers for x in numbers],dtype=float)
  return data.reshape(grid_shape)
# end def xsf_datagrid_3d_density

def wavefront_obj(verts,faces,normals):
  """ save polygons in obj format

  obj format is more commonly used than ply

  Args:
    verts (np.array): shape=(nvert,3) dtype=float, vertices in cartesian coordinates.
    faces (np.array): shape=(nface,nside) dtype=int, polygons each specified as a list of vertices (in vertex coordinates defined by verts).
    normals (np.array): shape=(nvert,3) dtype=float, normal vectors used for smooth lighting. There is one normal vector per vertex. 
  Returns:
    str: content of the obj file
  """
  text = ''
  if faces.dtype.kind != 'i':
    print('Warning: faces should be integers. Converting now.')
    faces = faces.astype(int)

  vert_fmt = '{name:s} {x:7.6f} {y:7.6f} {z:7.6f}\n' # weights not supported
  for ivert in range(len(verts)):
    vert  = verts[ivert]
    x,y,z = vert
    text += vert_fmt.format(name='v',x=x,y=y,z=z)
  # end for

  for inorm in range(len(normals)):
    norm  = normals[inorm]
    x,y,z = norm
    text += vert_fmt.format(name='vn',x=x,y=y,z=z)
  # end for inorm

  face_fmt = '{name:s} {polyx:d}//{normalx:d} {polyy:d}//{normaly:d} {polyz:d}//{normalz:d}\n' # texture not supported
  for iface in range(len(faces)):
    face  = faces[iface]
    x,y,z = face+1
    text += face_fmt.format(name='f',polyx=x,polyy=y,polyz=z,
      normalx=x,normaly=y,normalz=z)
  # end for iface

  return text
# end def wavefront_obj

def stanford_ply(verts,faces):
  """ save polygons in ply format

  ply is simpler than obj, but older and less used

  Args:
    verts (np.array): shape=(nvert,3) dtype=float, vertices in cartesian coordinates.
    faces (np.array): shape=(nface,nside) dtype=int, polygons each specified as a list of vertices (in vertex coordinates defined by verts).
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
  nface= len(faces)
  new_faces = np.zeros([nface,ndim+1],dtype=int)
  new_faces[:,0] = 3
  new_faces[:,1:]= faces

  text = header.format(nvert=len(verts),nface=nface) + \
    arr2text(verts) + arr2text(new_faces).strip('\n')

  return text
# end def stanford_ply
