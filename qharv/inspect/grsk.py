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
