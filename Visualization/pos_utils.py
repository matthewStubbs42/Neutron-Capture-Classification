import numpy as np

row_remap=np.flip(np.arange(16))

"""
mapping PMT index to geometric location.
copied from John Walker: https://github.com/WatChMaL/NeutronGNN/blob/master/root_utils/pos_utils.py
"""

def num_modules():
    """Returns the total number of mPMT modules"""
    return 536

def num_pmts():
    """Returns the total number of PMTs"""
    return num_modules()*19

def module_index(pmt_index):
    """Returns the module number given the 0-indexed pmt number"""
    return pmt_index//19

def pmt_in_module_id(pmt_index):
    """Returns the pmt number within a 
    module given the 0-indexed pmt number"""
    return pmt_index%19

def is_barrel(module_index):
    """Returns True if module is in the Barrel"""
    return ( (module_index<320) | ((module_index>=408)&(module_index<448)) )

def is_bottom(module_index):
    """Returns True if module is in the bottom cap"""
    return ( (module_index>=320)&(module_index<408) )

def is_top(module_index):
    """Returns True if module is in the top cap"""
    return ( (module_index>=448)&(module_index<536) )