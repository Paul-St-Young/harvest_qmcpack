# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Example input sections to QMCPACK. Intended to be easily extendable.
#  To edit an example for your need, please refer to lxml documentation. There are extensive documentation regarding how to iterate through the children of a node, get and set attributes, as well as insert the node into a larger xml document. In particular, xpath expression are very useful for grabbing what you need from an xml node.

from qharv.seed import xml
from io import StringIO

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
  return xml.parse(text)

def wbyw_dmc():
  text = '''<qmc method="dmc" move="not_pbyp_or_whatever" checkpoint="0">
    <parameter name="usedrift">    yes  </parameter>
    <parameter name="blocks">       64  </parameter>
    <parameter name="steps">       200  </parameter>
    <parameter name="timestep">  0.002  </parameter>
  </qmc>'''
  return xml.parse(text)

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
  return xml.parse(text)

# ============================= <backflow> section =============================

def bcc54_static_backflow():
  text = '''<backflow>
          <transformation name="eHB" type="e-I" function="Bspline" source="ion0">
            <correlation elementType="H" cusp="0.0" size="8">
              <coefficients id="eHB" type="Array" optimize="yes">
                -0.318225205 -0.3118825218 -0.2590876403 -0.178897934 -0.09533888539 -0.0359722158 -0.01122793946 -0.003273977434
              </coefficients>
            </correlation>
          </transformation>
          <transformation name="eeB" type="e-e" function="Bspline">
            <correlation speciesA="u" speciesB="u" cusp="0.0" size="8">
              <coefficients id="uuB" type="Array" optimize="yes">
                0.01700338137 0.006141973191 0.008060045646 0.006826236155 0.005407519967 0.003248534432 0.001432034258 0.0004420389469
              </coefficients>
            </correlation>
            <correlation speciesA="u" speciesB="d" cusp="0.0" size="8">
              <coefficients id="udB" type="Array" optimize="yes">
                0.1398741119 0.07423895812 0.04099268295 0.02056532273 0.008676400493 0.003448772417 0.001237987267 0.00058476928
              </coefficients>
            </correlation>
          </transformation>
        </backflow>'''
  return xml.parse(text)

def bcc54_dynamic_backflow():
  text = '''<backflow>
          <transformation name="eeB" type="e-e" function="Bspline">
            <correlation speciesA="u" speciesB="u" cusp="0.0" size="8">
              <coefficients id="uuB" type="Array" optimize="yes">
                0.008192250982 0.007186126165 0.007069997288 0.006619743038 0.005364967914 0.003620424119 0.001785836249 0.0006686008907
              </coefficients>
            </correlation>
            <correlation speciesA="u" speciesB="d" cusp="0.0" size="8">
              <coefficients id="udB" type="Array" optimize="yes">
                0.1309903648 0.07432011251 0.03921962633 0.01894998984 0.007714708214 0.00293134854 0.00105448654 0.0003790266034
              </coefficients>
            </correlation>
            <correlation speciesA="u" speciesB="p" cusp="0.0" size="8">
              <coefficients id="eHB" type="Array" optimize="yes">
                -0.3403607503 -0.1877305961 -0.1185710192 -0.06331053465 -0.02732069912 -0.004471724758 0.005910809959 0.004033706511
              </coefficients>
            </correlation>
            <correlation link="up" speciesA="d" speciesB="p"/>
            <correlation link="uu" speciesA="d" speciesB="d"/>
            <correlation link="up" speciesA="p" speciesB="p"/>
            <!-- pp backflow does nothing if hartree_product wf is used for protons -->
          </transformation>
        </backflow>'''
  return xml.parse(text)
