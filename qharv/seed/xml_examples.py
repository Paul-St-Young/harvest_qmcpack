# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Example input sections to QMCPACK. Intended to be easily extendable.
#  To edit an example for your need, please refer to lxml documentation. There are extensive documentation regarding how to iterate through the children of a node, get and set attributes, as well as insert the node into a larger xml document. In particular, xpath expression are very useful for grabbing what you need from an xml node.

from qharv.seed import xml
from io import StringIO

def text2node(text):
  return xml.read( StringIO(text.decode()) ).getroot()

# ============================= <qmc> section =============================

def wbyw_vmc():
  text = '''<qmc method="vmc" move="not_pbyp_or_whatever" checkpoint="-1">
    <parameter name="usedrift">    yes  </parameter>
    <parameter name="warmupsteps"> 750  </parameter>
    <parameter name="blocks">       40  </parameter>
    <parameter name="steps">        30  </parameter>
    <parameter name="timestep">    0.1  </parameter>
    <parameter name="samples">     512  </parameter>
  </qmc>'''
  return text2node(text)

def wbyw_dmc():
  text = '''<qmc method="dmc" move="not_pbyp_or_whatever" checkpoint="0">
    <parameter name="usedrift">    yes  </parameter>
    <parameter name="blocks">       40  </parameter>
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
      <parameter name="blocks">         40  </parameter>
      <parameter name="warmupsteps">  1024  </parameter>
      <parameter name="timestep">      0.1  </parameter>
      <parameter name="samples">    512000  </parameter>
      <parameter name="usedrift">      yes  </parameter>
      <parameter name="steps">           5  </parameter>
      <parameter name="exp0">           -8  </parameter>
      <parameter name="bigchange">     5.0  </parameter>
      <parameter name="stepsize">     0.02  </parameter>
      <parameter name="alloweddifference">  0.0001  </parameter>
    </qmc>
  </loop>'''
  return text2node(text)
