# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Kyrt is a versatile fabric exclusive to the planet Florina of Sark.
# The fluorescent and mutable kyrt is ideal for artsy decorations.
# OK, this is a library of reasonable defaults for matplotlib figures.
# May this library restore elegance to your plots.

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ======================== library of defaults =========================
# expose some default colors for convenience
from matplotlib.cm import get_cmap
cmap   = get_cmap('viridis')
colors = cmap.colors  # 256 default colors
errorbar_style = {
  'cyq':{
    'linestyle':'none',         # do 1 thing
    'markersize':3.5,           # readable
    'markeredgecolor':'black',  # accentuate
    'markeredgewidth':0.3,
    'capsize':4,
    'elinewidth':0.5
   }
}

# ======================== level 0: basic color =========================


def get_cmap(name='viridis'):
  """ return color map by name

  Args:
    name (str, optional): name of color map, default 'viridis'
  Return:
    matplotlib.colors.ListedColormap: requested colormap
  """
  from matplotlib import cm
  cmap = cm.get_cmap(name)
  return cmap


def get_norm(vmin, vmax):
  """ return norm function for scalar in range (vmin, vmax)

  Args:
    vmin (float): value minimum
    vmax (float): value maximum
  Return:
    matplotlib.colors.Normalize: color normalization function
  """
  norm = plt.Normalize(vmin, vmax)
  return norm


def scalar_colormap(vmin, vmax, name='viridis'):
  """ return a function that maps a number to a color

  Args:
    vmin (float): minimum scalar value
    vmax (float): maximum scalar value
    name (str, optional): color map name, default is 'viridis'
  Return:
    function: float -> (float,)*4 RGBA color space
  """
  cmap = get_cmap(name)
  norm = get_norm(vmin, vmax)

  def v2c(v):  # function mapping value to color
    return cmap(norm(v))
  return v2c


def scalar_colorbar(vmin, vmax, name='viridis', **kwargs):
  """ return a colorbar for scalar_color_map()

  Args:
    vmin (float): minimum scalar value
    vmax (float): maximum scalar value
    name (str, optional): color map name, default is 'viridis'
  Return:
    matplotlib.colorbar.Colorbar: colorbar
  """
  cmap = get_cmap(name)
  norm = get_norm(vmin, vmax)
  # issue 3644
  sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cbar = plt.colorbar(sm, **kwargs)
  return cbar


# ======================== level 0: basic ax edits =========================


def set_xy_format(ax, xfmt='%3.2f', yfmt='%3.2f'):
  """ change x,y tick formats e.g. number of digits

  Args:
    ax (plt.Axes): matplotlib axes
    xfmt (int,optional): xtick format, default is '%3.2f'
    yfmt (int,optional): ytick format, default is '%3.2f'
  """
  ax.get_xaxis().set_major_formatter( FormatStrFormatter(xfmt) )
  ax.get_yaxis().set_major_formatter( FormatStrFormatter(yfmt) )
# end def


def set_tick_font(ax, xsize=14, ysize=14,
  xweight='bold', yweight='bold', **kwargs):
  """ change x,y tick fonts

  Args:
    ax (plt.Axes): matplotlib axes
    xsize (int,optional): xtick fontsize, default is 14
    ysize (int,optional): ytick fontsize, default is 14
    xweight (str,optional): xtick fontweight, default is 'bold'
    yweight (str,optional): ytick fontweight, default is 'bold'
    kwargs (dict): other tick-related properties
  """
  plt.setp(ax.get_xticklabels(), fontsize=xsize,
    fontweight=xweight,**kwargs)
  plt.setp(ax.get_yticklabels(), fontsize=ysize,
    fontweight=yweight,**kwargs)


def set_label_font(ax, xsize=14, ysize=14,
  xweight='bold', yweight='bold', **kwargs):
  """ change x,y label fonts

  Args:
    ax (plt.Axes): matplotlib axes
    xsize (int,optional): xlabel fontsize, default is 14
    ysize (int,optional): ylabel fontsize, default is 14
    xweight (str,optional): xlabel fontweight, default is 'bold'
    yweight (str,optional): ylabel fontweight, default is 'bold'
    kwargs (dict): other label-related properties
  """
  plt.setp(ax.xaxis.label, fontsize=xsize,
    fontweight=xweight,**kwargs)
  plt.setp(ax.yaxis.label, fontsize=ysize,
    fontweight=yweight,**kwargs)


# ====================== level 0: basic legend edits =======================


def set_legend_marker_size(leg, ms=10):
  handl = leg.legendHandles
  msl   = [ms]*len(handl)  # override marker sizes here
  for hand,ms in zip(handl, msl):
    hand._legmarker.set_markersize(ms)


# ===================== level 1: interpolate scatter ======================

def show_spline(ax, line, spl_kws=dict(), nx=1024, **kwargs):
  """ show a smooth spline through given line x y

  Args:
    ax (plt.Axes): matplotlib axes
    line (Line1D): matplotlib line object
    spl_kws (dict, optional): keyword arguments to splrep, default is empty
    nx (int, optional): number of points to allocate to 1D grid
  Return:
    Line1D: interpolating line
  """
  import numpy as np
  from scipy.interpolate import splrep, splev
  myx = line.get_xdata()
  myy = line.get_ydata()
  color = line.get_color()
  finex = np.linspace(min(myx), max(myx), nx)
  tck = splrep(myx, myy, **spl_kws)
  line1 = ax.plot(finex, splev(finex, tck), c=color, **kwargs)
  return line1

# ======================== composition =========================


def pretty_up(ax):
  set_tick_font(ax)
  set_label_font(ax)
