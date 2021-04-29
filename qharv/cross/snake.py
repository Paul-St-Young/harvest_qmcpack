# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Routines to facilitate snakemake workflow.
def hybrid_expand(regex, zips, **regs):
  from snakemake.io import expand
  small = expand(regex, zip, **zips, allow_missing=True)
  big = expand(small, **regs)
  return big
