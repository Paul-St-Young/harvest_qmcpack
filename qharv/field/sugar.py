# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Python candy i.e. decorators
import os
import subprocess as sp

def check_dir_before(mkdir):
  def wrapper(dirname):
    if not os.path.isdir(dirname):
      mkdir(dirname)
  return wrapper

@check_dir_before
def mkdir(x):
  sp.check_call(['mkdir', '-p', x])

def skip_exist_file(write_file):
  def wrapper(fout, *args, **kwargs):
    if not os.path.isfile(fout):
      return write_file(fout, *args, **kwargs)
    else:
      print('%s exists' % fout)
  return wrapper

def cache(write_file):
  def wrapper(fout, *args, **kwargs):
    cache_dir = os.path.dirname(fout)
    if cache_dir != '':
      mkdir(cache_dir)
    return skip_exist_file(write_file)(fout, *args, **kwargs)
  return wrapper

def time(func):
  from time import time
  def wrapper(*args, **kwargs):
    tick = time()
    ret = func(*args, **kwargs)
    tock = time()
    dt = tock-tick
    return dt, ret
  return wrapper

def get_progress_bar(maxval):
  from progressbar import ProgressBar
  bar = ProgressBar(maxval=maxval)
  import sys
  if sys.version_info >= (3, 0):
    from progressbar import Bar, ETA
    widgets = [Bar('>'), ETA()]
    bar.widgets = widgets
    bar.start()
  return bar

def show_h5progress(collect_h5file):
  def wrapper(h5file, flist, *args, **kwargs):
    bar = get_progress_bar(len(flist))
    ifile = 0
    for ifile, floc in enumerate(flist):
      collect_h5file(h5file, floc, *args, **kwargs)
      bar.update(ifile)
  return wrapper

def concat_return(show_progress=True, fault_tolerant=False):
  """Show progress of concat reduce function"""
  def _concat_return(collect):
    """Concatenate the return value of a collect function on a file to a list

    Args:
      collect (callable): a function that parses a file into an object
    Return:
      list: a list of return values each from applying collect to a file
    """
    def wrapper(flist, *args, **kwargs):
      if show_progress:
        bar = get_progress_bar(len(flist))
      ifile = 0
      data = []
      for ifile, floc in enumerate(flist):
        try:
          result = collect(floc, *args, **kwargs)
          data.append(result)
        except Exception as err:
          if fault_tolerant:
            msg = str(err) + ' at:\n' + floc
            print(msg)
          else:
            raise err
        if show_progress:
          bar.update(ifile)
      return data
    return wrapper
  return _concat_return

@cache
def savefig(fig_loc, fig):
  fig.savefig(fig_loc, bbox_inches='tight', dpi=320)
