# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to collect QMCPACK data
#  pick the fruit of qharv.reel, qharv.sieve's labor
import os
import numpy as np
import pandas as pd

def get_kvecs_ym_ye(entry, yname, xname='kvecs'):
  ymean = '%s_mean' % yname
  yerror = '%s_error' % yname

  kvecs = np.array(entry[xname])
  ym = np.array(entry[ymean])
  ye = np.array(entry[yerror])

  return kvecs, ym, ye
