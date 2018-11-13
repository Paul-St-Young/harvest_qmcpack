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


def check_file_before(write_file):
  """ check if file exists before calling a function that overwrites the file """
  def wrapper(fout, *args, **kwargs):
    if not os.path.isfile(fout):
      write_file(fout, *args, **kwargs)
    else:
      raise RuntimeError('%s exists'%fout)
  return wrapper


def skip_exist_file(write_file):
  def wrapper(fout, *args, **kwargs):
    if not os.path.isfile(fout):
      write_file(fout, *args, **kwargs)
    else:
      print('%s exists'%fout)
  return wrapper

def show_h5progress(collect_h5file):
  def wrapper(h5file, flist, *args, **kwargs):
    from progressbar import ProgressBar
    bar = ProgressBar(maxval=len(flist))
    ifile = 0
    for ifile, floc in enumerate(flist):
      collect_h5file(h5file, floc, *args, **kwargs)
      bar.update(ifile)
  return wrapper
