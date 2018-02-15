# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Kyrt is a versatile fabric exclusive to the planet Florina of Sark.
# The fluorescent and mutable kyrt is ideal for artsy decorations.
# OK, this is a library of reasonable defaults for matplotlib figures.
# May this library restore elegance to your plots.

import matplotlib as mpl
import matplotlib.pyplot as plt

# expose some default colors for convenience
from matplotlib.cm import get_cmap
cmap   = get_cmap('viridis')
colors = cmap.colors # 256 default colors

# ======================== level 0: basic ax edits =========================
from matplotlib.ticker import FormatStrFormatter
def set_xy_format(ax,xfmt='%3.2f',yfmt='%3.2f'):
  """ change x,y tick formats e.g. number of digits
  Args:
    ax (plt.Axes): matplotlib axes
    xfmt (int,optional): xtick format, default is '%3.2f'
    yfmt (int,optional): ytick format, default is '%3.2f'
  """
  ax.get_xaxis().set_major_formatter( FormatStrFormatter(xfmt) )
  ax.get_yaxis().set_major_formatter( FormatStrFormatter(yfmt) )
# end def

def set_tick_font(ax,xsize=14,ysize=14
  ,xweight='bold',yweight='bold',**kwargs):
  """ change x,y tick fonts
  Args:
    ax (plt.Axes): matplotlib axes
    xsize (int,optional): xtick fontsize, default is 14
    ysize (int,optional): ytick fontsize, default is 14
    xweight (str,optional): xtick fontweight, default is 'bold'
    yweight (str,optional): ytick fontweight, default is 'bold'
    kwargs (dict): other tick-related properties
  """
  plt.setp(ax.get_xticklabels(),fontsize=xsize
    ,fontweight=xweight,**kwargs)
  plt.setp(ax.get_yticklabels(),fontsize=ysize
    ,fontweight=yweight,**kwargs)
# end def

def set_label_font(ax,xsize=14,ysize=14
  ,xweight='bold',yweight='bold',**kwargs):
  """ change x,y label fonts
  Args:
    ax (plt.Axes): matplotlib axes
    xsize (int,optional): xlabel fontsize, default is 14
    ysize (int,optional): ylabel fontsize, default is 14
    xweight (str,optional): xlabel fontweight, default is 'bold'
    yweight (str,optional): ylabel fontweight, default is 'bold'
    kwargs (dict): other label-related properties
  """
  plt.setp(ax.xaxis.label,fontsize=xsize
    ,fontweight=xweight,**kwargs)
  plt.setp(ax.yaxis.label,fontsize=ysize
    ,fontweight=yweight,**kwargs)
# end def

# ======================== composition =========================
def pretty_up(ax):
  set_tick_font(ax)
  set_label_font(ax)
# end def pretty_up
