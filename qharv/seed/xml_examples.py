# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Example input sections to QMCPACK. Intended to be easily extendable.
#  To edit an example for your need, please refer to lxml documentation. There are extensive documentation regarding how to iterate through the children of a node, get and set attributes, as well as insert the node into a larger xml document. In particular, xpath expression are very useful for grabbing what you need from an xml node.

from qharv.seed import xml
from io import StringIO

# ============================= <qmc> section =============================

def wbyw_vmc():
  text = '''<qmc method="vmc" move="not_pbyp_or_whatever" checkpoint="-1">
    <parameter name="usedrift">    yes      </parameter>
    <parameter name="warmupsteps">     512  </parameter>
    <parameter name="warmuptimestep"> 0.01  </parameter>
    <parameter name="blocks">       64  </parameter>
    <parameter name="steps">         4  </parameter>
    <parameter name="substeps">      4  </parameter>
    <parameter name="timestep">    0.05 </parameter>
    <parameter name="samples">     576  </parameter>
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
      <parameter name="blocks">          64  </parameter>
      <parameter name="warmupsteps">    512  </parameter>
      <parameter name="warmuptimestep">0.01  </parameter>
      <parameter name="timestep">      0.05  </parameter>
      <parameter name="substeps">         5  </parameter>
      <parameter name="samples">      49152  </parameter>
      <parameter name="usedrift">       yes  </parameter>
      <parameter name="MinMethod"> OneShiftOnly </parameter>
    </qmc>
  </loop>'''
  return xml.parse(text)

# ============================= <jastrow> section =============================
def i4_dynamic_jastrow():
  # optimized at rs=1.21 ca=2.50
  text = '''<jastrow type="Two-Body" name="J2" function="bspline" print="yes">
        <correlation speciesA="u" speciesB="u" size="8">
          <coefficients id="uu" type="Array"> 0.3982354817 0.2833633661 0.1990323682 0.133722635 0.08434542192 0.04923531445 0.02441131493 0.01008771115</coefficients>
        </correlation>
        <correlation speciesA="u" speciesB="d" size="8">
          <coefficients id="ud" type="Array"> 0.5900223267 0.3860433288 0.24311222 0.148586686 0.08958602691 0.05199246535 0.02648123887 0.01131666741</coefficients>
        </correlation>
        <correlation speciesA="u" speciesB="p" size="8" cusp="1.0">
          <coefficients id="up" type="Array"> -1.111746343 -0.6935590088 -0.4066043669 -0.2267990655 -0.1192950931 -0.05545092606 -0.01749765916 -0.005502643809</coefficients>
        </correlation>
        <correlation link="up" speciesA="d" speciesB="p"/>
        <correlation link="uu" speciesA="d" speciesB="d"/>
        <correlation speciesA="p" speciesB="p" size="8" cusp="0.0">
          <coefficients id="ppJ" type="Array"> 0.0003275129953 0.4180095185 1.160116171 0.1486270126 -0.4008289559 -0.400098892 -0.2304413008 -0.037986222</coefficients>
        </correlation>
      </jastrow>'''
  return xml.parse(text)

# ============================= <backflow> section =============================

def bcc54_static_backflow():
  # optimized at rs=1.31
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
  # optimized at rs=1.31
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

# ============================= <hamiltonian> section =============================

def static_ae_ham():
  text = '''<hamiltonian name="h0" type="generic" target="e">
         <pairpot type="coulomb" name="ElecElec" source="e" target="e"/>
         <pairpot type="coulomb" name="IonIon" source="ion0" target="ion0"/>
         <pairpot type="coulomb" name="ElecIon" source="ion0" target="e"/>
         <estimator name="csk" type="csk" hdf5="yes"/>
         <estimator type="gofr" name="gofr" num_bin="128"/>
         <estimator type="Pressure"/>
      </hamiltonian>'''
  return xml.parse(text)

def dynamic_ae_ham():
  text = '''<hamiltonian name="h0" type="generic" target="e">
         <pairpot type="coulomb" name="ElecElec" source="e" target="e"/>
         <estimator name="csk" type="csk" hdf5="yes"/>
         <estimator type="gofr" name="gofr" num_bin="128"/>
         <estimator type="Pressure"/>
         <estimator name="skinetic" type="specieskinetic"/>
         <estimator hdf5="yes" name="latdev" per_xyz="yes" sgroup="H" source="wf_centers" target="e" tgroup="p" type="latticedeviation"/>
      </hamiltonian>'''
  return xml.parse(text)

# =========================== <qmcsystem> section ===========================
def heg_system(rs, nshell_up, polarized):
  """ construct QMCPACK input xml <qmcsystem> node for the homogeneous
  electron gas (HEG). Momentum shells must be fully filled. The HEG must
  either be fully polarized or fully unpolarized.

  number of particles is determined by the number of filled shells.
   0 ->  1
   1 ->  7
   2 -> 19
   3 -> 27
   4 -> 33

  e.g. heg_system(1.0, 2, False) for unpolarized HEG with 2 shells
  (38 electrons) at rs=1.0

  * note: this function aims to return the **simplest** functional input,
  for more flexibility please edit the returned xml element instead of
  adding to this function.

  Args:
    rs (float): Wigner-Seitz radius
    nshell_up (int): number k shells filled by up electrons
    polarized (bool): True: polarized (ndn=0); False: unpolarized (nup=ndn)
  Return:
    lxml.etree.Element: qsys_node containing the <qmcsystem> xml node
  """
  from lxml import etree
  from copy import deepcopy

  # !!!! hard-code the number of electrons at each shell filling
  nshell2nelec = {
   -1:0,  # hack to remove electron species
    0:1,
    1:7,
    2:19,
    3:27,
    4:33,
    5:57,
    6:81,
    7:93,
    8:123,
    9:147,
    10:171
  }
  # check inputs
  if polarized:
    nshell_dn = -1
  else:
    nshell_dn = nshell_up

  avail_nshell = nshell2nelec.keys()
  if max(nshell_up, nshell_dn) > max(avail_nshell):
    raise RuntimeError('add to nshell2nelec; see example in HEGGrid.h')
  if nshell_up not in avail_nshell:
    raise RuntimeError('up shell not fully filled')
  if nshell_dn not in avail_nshell:
    raise RuntimeError('down shell not fully filled')

  # calculate and check polarization
  nup = nshell2nelec[nshell_up]
  ndn = nshell2nelec[nshell_dn]
  nelec = nup + ndn
  pol = int(round( abs(nup-ndn)/nelec ))
  assert pol == int(polarized)

  # build simulation cell using rs and nshell
  sc_node = etree.Element('simulationcell')
  rs_node = etree.Element('parameter', {
    'name':'rs'
    , 'condition': str(nelec)
    , 'polarized': str( int(polarized) )
  })
  rs_node.text = ' ' + str(rs) + ' '
  bc_node = etree.Element('parameter', {'name':'bconds'})
  bc_node.text = ' p p p '
  lr_node = etree.Element('parameter', {'name':'LR_dim_cutoff'})
  lr_node.text = ' 15.0 '
  for node in [rs_node, bc_node, lr_node]:
    sc_node.append(node)

  # build particleset node
  pset_node = etree.Element('particleset',{'name':'e', 'random':'yes'})
  ugrp_node = etree.Element('group',{'name':'u','size':str(nup)})
  dgrp_node = etree.Element('group',{'name':'d','size':str(ndn)})
  charge_node = etree.Element('parameter',{'name':'charge'})
  charge_node.text = ' -1 '
  for grp in [ugrp_node, dgrp_node]:
    grp.append( deepcopy(charge_node) )
    pset_node.append(grp)

  # build wavefunction node
  wf_node  = etree.Element('wavefunction', {'name':'psi0', 'target':'e'})
  det_node = etree.Element('determinantset',{
    'type':'electron-gas'
    , 'shell':str(nshell_up)
    , 'shell2':str(nshell_dn)
  })
  wf_node.append(det_node)

  # build hamiltonian node
  ham_node = etree.Element('hamiltonian', {
    'name':'h0'
    , 'type':'generic'
    , 'target':'e'
  })
  pot_node = etree.Element('pairpot', {
    'name':'ElecElec'
    , 'type':'coulomb'
    , 'source':'e'
    , 'target':'e'
  })
  ham_node.append(pot_node)

  # assemble qmcsystem
  qsys_node = etree.Element('qmcsystem')
  for node in [sc_node, pset_node, wf_node, ham_node]:
    qsys_node.append(node)

  return qsys_node
