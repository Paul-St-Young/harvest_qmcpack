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
def figaxad(figsize=None, labelsize=12, sharex=True):
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
  fig = plt.figure(figsize=figsize)
  axa = fig.add_subplot(gs[0:3, :])
  kws = dict()
  if sharex:
    kws['sharex'] = axa
  axd = fig.add_subplot(gs[3, :], **kws)
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

# ======================= level 1: advanced ax edits ========================

def cox(ax, x, xtlabels, **kwargs):
  """Add co-xticklabels at top of the plot, e.g., with a different unit

  Args:
    ax (plt.Axes): matplotlib axes
    x (list): xtick locations
    xtlabels (list): xtick labels
  """
  ax1 = ax.twiny()
  ax1.set_xlim(ax.get_xlim())
  ax.set_xticks(x)
  ax1.set_xticks(x)
  ax1.set_xticklabels(xtlabels, **kwargs)
  xtop(ax1)
  return ax1

def coy(ax, y, ytlabels):
  """Add co-yticklabels on the right of the plot, e.g., with a different unit

  Args:
    ax (plt.Axes): matplotlib axes
    y (list): ytick locations
    ytlabels (list): ytick labels
  """
  ax1 = ax.twinx()
  ax1.set_ylim(ax.get_ylim())
  ax.set_yticks(y)
  ax1.set_yticks(y)
  ax1.set_yticklabels(ytlabels)
  yright(ax1)
  return ax1

def align_ylim(ax1, ax2):
  ylim1 = ax1.get_ylim()
  ylim2 = ax2.get_ylim()
  ymin = min(ylim1[0], ylim2[0])
  ymax = max(ylim1[1], ylim2[1])
  ylim = (ymin, ymax)
  ax1.set_ylim(ylim)
  ax2.set_ylim(ylim)

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

# ====================== level 0: global edits =======================
def set_style(style='ticks', context='talk', **kwargs):
  import seaborn as sns
  if (context=='talk') and ('font_scale' not in kwargs):
    kwargs['font_scale'] = 0.9
  sns.set_style(style)
  sns.set_context(context, **kwargs)

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

# ======================== level 0: basic patches =========================
def show_circle(ax, radius, center=None, **kwargs):
  if 'fill' not in kwargs:
    kwargs['fill'] = False
  if 'color' not in kwargs:
    kwargs['color'] = 'k'
  if center is None:
    center = (0, 0)
  circ = plt.Circle(center, radius, **kwargs)
  ax.add_patch(circ)

# ===================== level 1: fit line ======================
def show_fit(ax, line, model=None, sel=None, nx=64, popt=None,
  xmin=None, xmax=None, circle=True, circle_style=None,
  cross=False, cross_style=None, **kwargs):
  """ fit a segment of (x, y) data and show fit

  get x, y data from line; use sel to make selection

  Args:
    ax (Axes): matplotlib axes
    line (Line2D): line with data
    model (callable, optional): model function, default linear function
    sel (np.array, optional): boolean selector array
    nx (int, optional): grid size, default 64
    xmin (float, optional): grid min
    xmax (float, optional): grid max
    circle (bool, optional): circle selected points, default True
    cross (bool, optional): cross out deselected points, default False
  Return:
    (np.array, np.array, list): (popt, perr, lines)
  """
  import numpy as np
  from scipy.optimize import curve_fit
  # get and select data to fit
  myx = line.get_xdata()
  myy = line.get_ydata()
  # default to linear fit
  if model is None:
    def model(x, a, b):
      return a+b*x
  # show selected data
  if sel is None:
    sel = np.ones(len(myx), dtype=bool)
  myx1 = myx[sel]
  myy1 = myy[sel]
  myx11 = myx[~sel]
  myy11 = myy[~sel]
  if xmin is None:
    xmin = myx1.min()
  if xmax is None:
    xmax = myx1.max()
  lines = []
  if circle:
    styles = get_style(line)
    styles['linestyle'] = ''
    styles['marker'] = 'o'
    styles['fillstyle'] = 'none'
    if circle_style is not None:
      styles.update(circle_style)
    line1 = ax.plot(myx[sel], myy[sel], **styles)
    lines.append(line1[0])
  if cross:
    styles = get_style(line)
    styles['linestyle'] = ''
    styles['marker'] = 'x'
    if cross_style is not None:
      styles.update(cross_style)
    line11 = ax.plot(myx11, myy11, **styles)
    lines.append(line11[0])
  if popt is None:  # perform fit
    popt, pcov = curve_fit(model, myx1, myy1)
    perr = np.sqrt(np.diag(pcov))
  else:
    perr = None
  # show fit
  finex = np.linspace(xmin, xmax, nx)
  line2 = ax.plot(finex, model(finex, *popt),
    c=line.get_color(), **kwargs)
  lines.append(line2[0])
  return popt, perr, lines

def smooth_bspline(myx, myy, nxmult=10, **spl_kws):
  import numpy as np
  from scipy.interpolate import splrep, splev
  nx = len(myx)*nxmult
  idx = np.argsort(myx)
  tck = splrep(myx[idx], myy[idx], **spl_kws)
  finex = np.linspace(myx.min(), myx.max(), nx)
  finey = splev(finex, tck)
  return finex, finey

def show_spline(ax, line, spl_kws=dict(), sel=None, **kwargs):
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
  myx = line.get_xdata()
  myy = line.get_ydata()
  if sel is None:
    sel = np.ones(len(myx), dtype=bool)
  myx = myx[sel]
  myy = myy[sel]
  finex, finey = smooth_bspline(myx, myy, **spl_kws)
  color = line.get_color()
  line1 = ax.plot(finex, finey, c=color, **kwargs)
  return line1

def krig(finex, x0, y0, length_scale, noise_level):
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import DotProduct, RBF
  from sklearn.gaussian_process.kernels import WhiteKernel
  kernel = DotProduct() + RBF(length_scale=length_scale)
  kernel += WhiteKernel(noise_level=noise_level)
  gpr = GaussianProcessRegressor(kernel=kernel)
  gpr.fit(x0[:, None], y0)
  ym, ye = gpr.predict(finex[:, None], return_std=True)
  return ym, ye

def gpr_errorshade(ax, x, ym, ye,
  length_scale, noise_level, fb_kwargs=None,
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
  if fb_kwargs is None:
    fb_kwargs = {'color': myc, 'alpha': 0.4}
  eline = ax.fill_between(finex, ylm-yle, yhm+yhe, **fb_kwargs)
  return line[0], eline

# ===================== level 2: contour ======================
def contour_scatter(ax, xy, z, zlim=None, nz=8, cmap='viridis',
  interp_method='linear', mesh=(32, 32), lims=None, **kwargs):
  """View sampled scalar field using contours

  Args:
    ax (plt.Axes): matplotlib axes
    xy (np.array): scatter points, a list of 2D vectors
    z (np.array): scatter values, one at each scatter point
    zlim (list, optional): value (min, max) for colormap
    cmap (str, optional): color map name, default is 'viridis'
    nz (int, optional): number of contour lines for when zlim is set
    interp_method (str, optional): griddata, default 'linear'
    mesh (tuple, optional): regular grid shape, default (32, 32)
    kwargs (dict, optional): keyword arguments to be passed to ax.scatter
  Returns:
    matplotlib.contour.QuadContourSet: filled contour plot
  Example:
    >>> kxy = kvecs[:, :2]
    >>> nkxy = nofk[:, :, 0]
    >>> cs = contour_scatter(ax, kxy, nkxy, zlim=(0, 2), mesh=(256, 256))
  """
  import numpy as np
  from scipy.interpolate import griddata
  # interpret inputs and set defaults
  if zlim is not None:
    levels = np.linspace(*zlim, nz)
    if 'levels' in kwargs:
      msg = 'multiple values for keyward argument \'levels\''
      raise TypeError(msg)
    kwargs['levels'] = levels
  if lims is None:
    xarr = xy[:, 0]
    xmin = xarr.min()
    xmax = xarr.max()
    yarr = xy[:, 1]
    ymin = yarr.min()
    ymax = yarr.max()
    lims = ((xmin, xmax), (ymin, ymax))
  # create regular grid
  finex = np.linspace(lims[0][0], lims[0][1], mesh[0])
  finey = np.linspace(lims[1][0], lims[1][1], mesh[1])
  fine_points = [[(x, y) for y in finey] for x in finex]
  # interpolate scatter on regular grid
  interp_data = griddata(xy, z, fine_points, method=interp_method)
  finez = interp_data.reshape(*mesh).T
  # make contour plot
  cs = ax.contourf(finex, finey, finez, cmap=cmap, **kwargs)
  return cs

def color_scatter(ax, xy, z, zlim=None, cmap='viridis',
  **kwargs):
  """View sampled scalar field using value as color

  Args:
    ax (plt.Axes or Axes3D): matplotlib axes
    xy (np.array): scatter points, a list of 2D or 3D vectors
    z (np.array): scatter values, one at each scatter point
    zlim (list, optional): value (min, max) for colormap
    cmap (str, optional): color map name, default is 'viridis'
    kwargs (dict, optional): keyword arguments to be passed to ax.scatter
  Returns:
    mpl_toolkits.mplot3d.art3d.Path3DCollection: scatter plot
  Example:
    >>> s = color_scatter(ax, kvecs, nofk, zlim=(0, 2))
  """
  if zlim is None:
    zlim = (z.min(), z.max())
  v2c = scalar_colormap(*zlim, cmap)
  s = ax.scatter(*xy.T, c=v2c(z), **kwargs)
  return s

def tile_scatter(rvecs, mesht, axes=None):
  """Tile scatter data

  Args:
    rvecs (np.array): grid points
    mesht (tuple): tile mesh
    axes (np.array, optional): lattice vectors
  Example:
    tile charge density data (rvecs, rho) by 2x2
    >>> axes = np.diag([1.5, 2.0]); mesht = (2, 2)
    >>> rvecs = tile_scatter(rvecs, mesht, axes)
    >>> rho = tile_scatter(rho, mesht)
  """
  import numpy as np
  if axes is not None:
    from qharv.seed.hamwf_h5 import get_kvecs
    shifts = get_kvecs(axes, mesht)
    xl = []
    for shift in shifts:
      xl.append(rvecs+shift)
  else:  # no shift
    xl = [rvecs for i in range(np.prod(mesht))]
  x = np.concatenate(xl, axis=0)
  return x

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
