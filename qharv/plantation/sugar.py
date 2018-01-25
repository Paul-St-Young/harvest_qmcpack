# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Python candy i.e. decorators
import os

def check_dir_before(mkdir):
  def wrapper(dirname):
    if not os.path.isdir(dirname):
      mkdir(dirname)
  return wrapper

mkdir = lambda x:check_dir_before( os.mkdir )
