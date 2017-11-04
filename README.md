![master build status](https://api.travis-ci.org/Paul-St-Young/harvest_qmcpack.svg?branch=master)
# harvest_qmcpack
Python module containing useful routines to inspect and modify qmcpack objects.

## Quick Start

### Install
Clone the repository and add it to PYTHONPATH. To use examples, add bin to PATH.
```shell
git clone https://github.com/Paul-St-Young/harvest_qmcpack.git ~
export PYTHONPATH=~/harvest_qmcpack:$PYTHONPATH
export PATH=~/harvest_qmcpack/bin:$PATH
```

You can also use pip if you do not intend to change the code
```shell
git clone https://github.com/Paul-St-Young/harvest_qmcpack.git ~/harvest_qmcpack
pip install --user ~/harvest_qmcpack
```
to update to the newest version:
```shell
cd ~/harvest_qmcpack
git pull
pip install --user --upgrade ~/harvest_qmcpack
```

### Use
The library functions can be used in a python script
```python
# extract all scalar data from a run directory
import os
from qharv.reel  import scalar_dat, mole
from qharv.sieve import scalar_df

# define equilibration length and autocorrelation length
nequil = 5
kappa  = 1.0 # None to re-calculate
#  runs should be designed to have short equilibration and
# no autocorrelation. kappa can be calculated on-the-fly
# ,however kappa calculation is slow. I have yet to find
# a fast and reliable algorithm for nequil.

# generate the list of scalar.dat files to analyze
flist = mole.files_scalar_dat('./runs')
#  hint: use `moles.files_with_regex` for more general situations

# analyze the list of scalar.dat files
data  = []
for floc in flist:
  mydf = scalar_dat.parse(floc)
  mdf  = scalar_df.mean_error_scalar_df(mydf,nequil,kappa=kappa)
  assert len(mdf) == 1 # each scalar.dat should contribute to one entry
  # add metadata to identify runs
  mdf['path'] = os.path.dirname(floc)
  mdf['fdat'] = os.path.basename(floc)
  data.append(mdf)
df = pd.concat(data)
```

The examples in the "bin" folder can be ran in the shell
```shell
stalk vmc.in.xml
```

### Requirements
Requirements can be installed without admin access using `pip install --user -r requirements.txt`.

### Documentation
Documentation is available on [github pages][doc html]. A local copy can be generated using sphinx (`pip install --user sphinx`).
To generate the documentation, first use sphinx-apidoc to convert doc strings to rst documentation:
```shell
cd ~/harvest_qmcpack/doc; sphinx-apidoc -o source ../qharv
```
Next, use the Makefile to create html documentation:
```shell
cd ~/harvest_qmcpack/doc; make html
```
Finally, use your favorite browser to view the documentation:
```shell
cd ~/harvest_qmcpack/doc/build; firefox index.html
```

### Examples
Example usage of the qharv library are included in the "harvest_qmcpack/bin" folder. Each file in the folder is a Python script that performs a very specific task:
* stab: Scalar TABle (stab) analyzer, analyze one column of a scalar table file, e.g. `stab vmc.s000.scalar.dat`
* rebuild_wf: Rerun QMCPACK on optimized wavefunctions, e.g. `rebuild_wf opt.xml`
* stalk: show crystal structure specified in a QMCPACK input e.g. `stalk vmc.in.xml`

### Description
This module is intended to speed up on-the-fly setup, run, and analysis of QMCPACK calculations. The module should be used as a collection of glorified bash commands, which are usable in Python.
This module is NOT intended to be a full-fledged workflow tool. Please refer to [nexus][nexus] for complete workflow magnagement.

[nexus]:http://qmcpack.org/nexus/
[doc html]: https://paul-st-young.github.io/harvest_qmcpack/
