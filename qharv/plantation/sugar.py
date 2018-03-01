# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Python candy i.e. decorators
import os

def check_dir_before(mkdir):
  def wrapper(dirname):
    if not os.path.isdir(dirname):
      mkdir(dirname)
  return wrapper

mkdir = lambda x:check_dir_before( os.mkdir )(x)

def check_file_before(write_file):
  """ check if file exists before calling a function that overwrites the file """
  def wrapper(fout,*args,**kwargs):
    if not os.path.isfile(fout):
      write_file(fout,*args,**kwargs)
    else:
      print('%s exists'%fout)
  return wrapper
