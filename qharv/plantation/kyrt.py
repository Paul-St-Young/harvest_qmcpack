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
dark8 = [  # Colors from www.ColorBrewer.org by Cynthia A. Brewer, Geography, Pennsylvania State University.
  '#1b9e77',
  '#d95f02',
  '#7570b3',
  '#e7298a',
  '#66a61e',
  '#e6ab02',
  '#a6761d',
  '#666666'
]
errorbar_style = {
  'cyq': {
    'linestyle': 'none',         # do 1 thing
    'markersize': 3.5,           # readable
    'markeredgecolor': 'black',  # accentuate
    'markeredgewidth': 0.3,
    'capsize': 4,
    'elinewidth': 0.5
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
def figaxad(labelsize=12):
  """ construct a absolute/difference (ad) figure
   top 3/4 of the plot will be comparison at an absolute scale
   bottom 1/4 of the plot will be comparison at a relative scale

  Args:
    labelsize (int, optional): tick label size
  Return:
    (fig, axa, axd): figure and axes for absolute and difference plots
  """
  from matplotlib.gridspec import GridSpec
  gs = GridSpec(4, 4)
  fig = plt.figure()
  axa = fig.add_subplot(gs[0:3, :])
  axd = fig.add_subplot(gs[3, :], sharex=axa)
  plt.setp(axa.get_xticklabels(), visible=False)
  axa.tick_params(axis='y', labelsize=labelsize)
  axd.tick_params(labelsize=labelsize)
  fig.subplots_adjust(hspace=0)
  return fig, axa, axd

def set_xy_format(ax, xfmt='%3.2f', yfmt='%3.2f'):
  """ change x,y tick formats e.g. number of digits

  Args:
    ax (plt.Axes): matplotlib axes
    xfmt (int,optional): xtick format, default is '%3.2f'
    yfmt (int,optional): ytick format, default is '%3.2f'
  """
  ax.get_xaxis().set_major_formatter(FormatStrFormatter(xfmt))
  ax.get_yaxis().set_major_formatter(FormatStrFormatter(yfmt))

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
    fontweight=xweight, **kwargs)
  plt.setp(ax.get_yticklabels(), fontsize=ysize,
    fontweight=yweight, **kwargs)

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
    fontweight=xweight, **kwargs)
  plt.setp(ax.yaxis.label, fontsize=ysize,
    fontweight=yweight, **kwargs)

def xtop(ax):
  """ move xaxis label and ticks to the top

  Args:
    ax (plt.Axes): matplotlib axes
  """
  xaxis = ax.get_xaxis()
  xaxis.tick_top()
  xaxis.set_label_position('top')

def yright(ax):
  """ move yaxis label and ticks to the right

  Args:
    ax (plt.Axes): matplotlib axes
  """
  yaxis = ax.get_yaxis()
  yaxis.tick_right()
  yaxis.set_label_position('right')

# ====================== level 0: basic legend edits =======================
def set_legend_marker_size(leg, ms=10):
  handl = leg.legendHandles
  msl   = [ms]*len(handl)  # override marker sizes here
  for hand, ms in zip(handl, msl):
    hand._legmarker.set_markersize(ms)

def create_legend(ax, styles, labels, **kwargs):
  """ create custom legend

  learned from "Composing Custom Legends"

  Args:
    ax (plt.Axes): matplotlib axes
  Return:
    plt.legend.Legend: legend artist
  """
  from matplotlib.lines import Line2D
  custom_lines = [Line2D([], [], **style) for style in styles]
  leg = ax.legend(custom_lines, labels, **kwargs)
  return leg

# ====================== level 0: basic Line2D edits =======================
def get_style(line):
  """ get plot styles from Line2D object

  mostly copied from "Line2D.update_from"

  Args:
    line (Line2D): source of style
  Return:
    dict: line styles readily usable for another plot
  """
  styles = {
    'linestyle': line.get_linestyle(),
    'linewidth': line.get_linewidth(),
    'color': line.get_color(),
    'markersize': line.get_markersize(),
    'linestyle': line.get_linestyle(),
    'marker': line.get_marker()
  }
  return styles

# ====================== level 0: basic Line2D =======================
def errorshade(ax, x, ym, ye, **kwargs):
  line = ax.plot(x, ym, **kwargs)
  alpha = 0.4
  myc = line[0].get_color()
  eline = ax.fill_between(x, ym-ye, ym+ye, color=myc, alpha=alpha)
  return line, eline

# ===================== level 1: fit line ======================
def show_fit(ax, line, model, sel=None, nx=64, xmin=None, xmax=None, **kwargs):
  """ fit a segment of (x, y) data and show fit

  get x, y data from line; use sel to make selection

  Args:
    ax (Axes): matplotlib axes
    line (Line2D): line with data
    model (callable): model function
    sel (np.array, optional): boolean selector array
    nx (int, optional): grid size, default 64
    xmin (float, optional): grid min
    xmax (float, optional): grid max
  Return:
    (np.array, np.array, list): (popt, perr, lines)
  """
  import numpy as np
  from scipy.optimize import curve_fit
  # get and select data to fit
  myx = line.get_xdata()
  myy = line.get_ydata()
  # show selected data
  if sel is None:
    sel = np.ones(len(myx), dtype=bool)
  myx1 = myx[sel]
  myy1 = myy[sel]
  if xmin is None:
    xmin = myx1.min()
  if xmax is None:
    xmax = myx1.max()
  styles = get_style(line)
  styles['ls'] = ''
  styles['marker'] = 'o'
  styles['fillstyle'] = 'none'
  line1 = ax.plot(myx[sel], myy[sel], **styles)
  # perform fit
  popt, pcov = curve_fit(model, myx1, myy1)
  perr = np.sqrt(np.diag(pcov))
  # show fit
  finex = np.linspace(xmin, xmax, nx)
  line2 = ax.plot(finex, model(finex, *popt),
    c=line.get_color(), **kwargs)
  lines = [line1[0], line2[0]]
  return popt, perr, lines

def show_spline(ax, line, spl_kws=dict(), nx=1024, sel=None, **kwargs):
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
  if sel is None:
    sel = np.ones(len(myx), dtype=bool)
  myx = myx[sel]
  myy = myy[sel]
  color = line.get_color()
  finex = np.linspace(min(myx), max(myx), nx)
  tck = splrep(myx, myy, **spl_kws)
  line1 = ax.plot(finex, splev(finex, tck), c=color, **kwargs)
  return line1

def krig(finex, x0, y0, length_scale, noise_level):
  from sklearn.gaussian_process.gpr import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import DotProduct, RBF
  from sklearn.gaussian_process.kernels import WhiteKernel
  kernel = DotProduct() + RBF(length_scale=length_scale)
  kernel += WhiteKernel(noise_level=noise_level)
  gpr = GaussianProcessRegressor(kernel=kernel)
  gpr.fit(x0[:, None], y0)
  ym, ye = gpr.predict(finex[:, None], return_std=True)
  return ym, ye

def gpr_errorshade(ax, x, ym, ye,
  length_scale, noise_level,
  **kwargs):
  """WARNING: length_scale and noise_level are VERY DIFFICULT to tune """
  # make errorbar plot and extract color
  if ('ls' not in kwargs) and ('linestyle' not in kwargs):
    kwargs['ls'] = ''
  line = ax.errorbar(x, ym, ye, **kwargs)
  myc = line[0].get_color()
  # smoothly fit data
  import numpy as np
  dx = abs(x[1]-x[0])
  xmin = x.min(); xmax = x.max()
  finex = np.arange(xmin, xmax, dx/10.)
  ylm, yle = krig(finex, x, ym-ye,
    length_scale=length_scale, noise_level=noise_level)
  yhm, yhe = krig(finex, x, ym+ye,
    length_scale=length_scale, noise_level=noise_level)
  # plot fit
  fb_kwargs = {'color': myc, 'alpha': 0.4}
  eline = ax.fill_between(finex, ylm-yle, yhm+yhe, **fb_kwargs)
  return line[0], eline

# ===================== level 2: insets ======================
def inset_zoom(fig, ax_box, xlim, ylim, draw_func, xy_label=False):
  """ show an inset that zooms into a given part of the figure

  Args:
    fig (plt.Figure): figure
    ax_box (tuple): inset location and size (x0, y0, dx, dy) in figure ratio
    xlim (tuple): (xmin, xmax)
    ylim (tuple): (ymin, ymax)
    draw_func (callable): draw_func(ax) should recreate the figure
    xy_label (bool, optional): label inset axes, default is False
  Return:
    plt.Axes: inset axes
  Example:
    >>> ax1 = inset_zoom(fig, [0.15, 0.15, 0.3, 0.3], [0.1, 0.5], [-0.02, 0.01],
    >>>                  lambda ax: ax.plot(x, y))
    >>> ax.indicate_inset_zoom(axins)
  """
  ax1 = fig.add_axes(ax_box)
  ax1.set_xlim(*xlim)
  ax1.set_ylim(*ylim)
  draw_func(ax1)
  if not xy_label:
    ax1.set_xticks([])
    ax1.set_yticks([])
  return ax1

# ======================== composition =========================
def pretty_up(ax):
  set_tick_font(ax)
  set_label_font(ax)
