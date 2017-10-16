# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Example input sections to QMCPACK. Intended to be easily extendable.
#  To edit an example for your need, please refer to lxml documentation. There are extensive documentation regarding how to iterate through the children of a node, get and set attributes, as well as insert the node into a larger xml document. In particular, xpath expression are very useful for grabbing what you need from an xml node.

from qharv.seed import xml
from io import StringIO

def text2node(text):
  return xml.read( StringIO(text.decode()) ).getroot()

def set_param(node,pname,pval):
  """ set <parameter> with name 'pname' to 'pval' """
  assert type(pval) is str
  pnode = node.find('.//parameter[@name="%s"]'%pname)
  pnode.text = pval
# end def

# ============================= <qmc> section =============================

def wbyw_vmc():
  text = '''<qmc method="vmc" move="not_pbyp_or_whatever" checkpoint="-1">
    <parameter name="usedrift">    yes      </parameter>
    <parameter name="warmupsteps">     750  </parameter>
    <parameter name="warmuptimestep"> 0.01  </parameter>
    <parameter name="blocks">       64  </parameter>
    <parameter name="steps">        16  </parameter>
    <parameter name="timestep">    0.08  </parameter>
    <parameter name="samples">     512  </parameter>
  </qmc>'''
  return text2node(text)

def wbyw_dmc():
  text = '''<qmc method="dmc" move="not_pbyp_or_whatever" checkpoint="0">
    <parameter name="usedrift">    yes  </parameter>
    <parameter name="blocks">       64  </parameter>
    <parameter name="steps">       200  </parameter>
    <parameter name="timestep">  0.002  </parameter>
  </qmc>'''
  return text2node(text)

def wbyw_optimize():
  text = '''<loop max="8">
    <qmc method="linear" move="not_pbyp_or_whatever" checkpoint="-1">
      <cost name="energy">                0.95  </cost>
      <cost name="unreweightedvariance">  0.00  </cost>
      <cost name="reweightedvariance">    0.05  </cost>
      <parameter name="usedrift">      yes  </parameter>
      <parameter name="warmupsteps">    1024  </parameter>
      <parameter name="warmuptimestep"> 0.01  </parameter>
      <parameter name="timestep">       0.08  </parameter>
      <parameter name="blocks">         64  </parameter>
      <parameter name="steps">           5  </parameter>
      <parameter name="samples">     65536  </parameter>
      <parameter name="MinMethod"> OneShiftOnly </parameter>
    </qmc>
  </loop>'''
  return text2node(text)

# ============================= <backflow> section =============================

def bcc54_backflow():
  text = '''<backflow>
  <transformation name="eHB" type="e-I" function="Bspline" source="ion0">
     <correlation elementType="H" cusp="0.0" size="8">
      <coefficients id="eHB" type="Array" optimize="yes"> -0.1607770658 -0.01312455519 -0.01096274521 -0.02064241065 -0.0163772626 -0.01206014211 -0.01003052047 -0.004077335794</coefficients>
     </correlation>
  </transformation>
  <transformation name="eeB" type="e-e" function="Bspline">
     <correlation speciesA="u" speciesB="u" cusp="0.0" size="8">
      <coefficients id="uuB" type="Array" optimize="yes"> 0.09613117178 0.0693866824 0.03958692855 0.0257978026 0.01445303622 0.007976855226 0.003592602563 0.001525251225</coefficients>
     </correlation>
     <correlation speciesA="u" speciesB="d" cusp="0.0" size="8">
      <coefficients id="udB" type="Array" optimize="yes"> 0.1754712467 0.1132199636 0.06953867095 0.04139840285 0.02288496026 0.01144408177 0.005019197667 0.002024292924</coefficients>
     </correlation>
  </transformation>
  </backflow>'''
  return text2node(text)
