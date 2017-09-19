# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to manipulate pyscf results for use in QMCPACK
import numpy as np

def atom_text(elem,pos):
  """convert elem,pos to text representation
  
  for example, elem = ['C','C'], pos = [[0,0,0],[0.5,0.5,0.5]] will be 
  converted to 'C 0 0 0;\nC0.5 0.5 0.5'
  
  Args:
   elem (list): a list of atomic symbols such as 'H','C','O'
   pos  (list): a list of atomic positions, assume in 3D
  Returns:
   str: atomic string accepted by pyscf"""
  assert len(elem) == len(pos)
  lines = []
  for iatom in range(len(elem)):
      mypos = pos[iatom]
      assert len(mypos) == 3
      line = '%5s  %10.6f  %10.6f  %10.6f' % (elem[iatom],mypos[0],mypos[1],mypos[2])
      lines.append(line)
  atext = ';\n'.join(lines)
  return atext
# end def

def mf_from_chkfile(chkfile,scf_class=None):
  from pyscf.pbc import scf
  if scf_class is None:
    scf_class = scf.RHF

  cell,scf_rec = scf.chkfile.load_scf(chkfile)
  mf = scf.RHF(cell)
  mf.__dict__.update(scf_rec)
  return mf
# end def

def pw_to_r(gvecs,psig,grid_shape=None):
  """ convert a 3D function from plane-wave to real-space basis
  
  plane wave basis is assumed to be in reciprocal lattice units
  real space basis will be in grid units

  Args:
    gvecs (np.array):  dtype=int & shape = (npw,ndim), npw is the number of plane waves, and ndim is the number of spatial dimensions. ndim is expected to be 3. Each entry in gvecs should be a 3D vector of integers.  Each gvec specify a plane wave basis element exp(i*gvec*rvec).
    psig (np.array): dtype=complex & shape = (npw,). Expansion coefficients in PW basis.
    grid_shape (np.array): dtype=int & shape = (ndim,). Shape of real-space grid.
  Returns:
    (np.array,np.array): (grid_shape,moR), grid_shape is input if given. Otherwise constructed in function to hold all information from the plane-wave representation. moR has dtype=complex & shape = grid_shape. moR is the 3D function in real-space basis. Typically a molecular orbital.
  """
  
  pw_grid_shape = gvecs.max(axis=0) - gvecs.min(axis=0) + 1
  if grid_shape is None: # deduce minimum real-space basis to retain all information
    grid_shape = pw_grid_shape
  else: # make sure no information is lost from pw representation
    if not (grid_shape >= pw_grid_shape).all():
      raise RuntimeError('Please increase grid_shape beyond (%s) to preserve all information from the plane-wave representation.' % ' '.join(pw_grid_shape.astype(str)) )
    # end if
  # end if

  # perform Fourier transform
  npw,ndim = gvecs.shape
  assert ndim == 3
  fftbox = np.zeros(grid_shape,dtype=complex)
  for ig in range(npw):
    fftbox[tuple(gvecs[ig])] = psig[ig]
  # end for ig
  rgrid = np.fft.ifftn(fftbox) * np.prod(grid_shape)
  return grid_shape,rgrid
# end def pw_to_r

def r_to_pw(moR0,grid_shape,gvecs=None):
  """ convert a 3D function from real-space to plane-wave basis
  
  This function is essentially the inverse of pw_to_r, but assumes that the real space grid is built around (0,0,0)

  Args:
    moR0 (np.array): dtype=complex & shape = (ngrid,), ngrid is the number of grid points. 3D function in real-space grid basis.
    grid_shape (np.array): dtype=int & shape = (ndim,). Shape of real-space grid.
  Returns:
    (np.array,np.array): (gvecs,psig), gvecs is input if given. Otherwise internally constructed. psig has dtype=complex & shape = (npw,). psig is the 3D function in plane-wave basis. Typically a molecular orbital.
  """
  assert np.prod(grid_shape) == len(moR0)
  cell_gs = (grid_shape-1)/2
  
  if gvecs is None: # deduce minimum plane-wave basis to retain all information
    if not (2*cell_gs+1==grid_shape).all():
      raise RuntimeError('Please provide grid_shape. I cannot deduce minimum plane-wave basis for even grid_shape %s'%' '.join(grid_shape.astype(str)))
    # end if

    nx,ny,nz = cell_gs
    from itertools import product
    gvecs = np.array([gvec for gvec in product(
      range(-nx,nx+1),range(-ny,ny+1),range(-nz,nz+1))],dtype=int)
  else: # make sure information is not missing from real-space representation
    assert (gvecs.dtype is int)
    valid = (gvecs.max(axis=0) <= cell_gs).all() and (-gvecs.min(axis=0) <= cell_gs).all()
    if not valid:
      raise RuntimeError('Please remove gvectors outside of cell_gs: %s' % ' '.join(cell_gs.astype(str)) )
    # end if
  # end if

  npw,ndim = gvecs.shape
    
  orb = moR0.reshape(grid_shape)
  moG = np.fft.fftn(orb)/np.prod(grid_shape)

  psig = np.zeros(npw,dtype=complex)
  for ipw in range(npw):
    psig[ipw] = moG[tuple(gvecs[ipw])]
  # end for ipw

  return gvecs,psig
# end def r_to_pw
