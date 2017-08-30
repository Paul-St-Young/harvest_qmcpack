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
git clone https://github.com/Paul-St-Young/harvest_qmcpack.git ~
pip install --user ~/harvest_qmcpack
```

### Use
```python
from qharv.reel import scalar_dat
df = scalar_dat.parse('vmc.s000.scalar.dat')
```

### Requirements
* lxml
* numpy
* pandas

Requirements can be installed without admin access using `pip install --user [package name]`.

### Documentation
Documentation can be generated using sphinx (`pip install --user sphinx`).
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

### Description
This module is intended to speed up on-the-fly setup, run, and analysis of QMCPACK calculations. The module should be used as a collection of glorified bash commands, which are usable in Python.
This module is NOT intended to be a full-fledged workflow tool. Please refer to [nexus][nexus] for complete workflow magnagement.

[nexus]:http://qmcpack.org/nexus/
