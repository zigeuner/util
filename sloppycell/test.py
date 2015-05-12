"""
Testing mcautil.py.
"""

from __future__ import division 
import unittest

from SloppyCell.ReactionNetworks import *

from util.sloppycell.mca import mcautil
reload(mcautil)

from models.ma1_rev import net as net1
from models.path import net as net_path


class TestSSConcns(unittest.TestCase):
    known = [(net_path, [4])]
    
    def test_known(self):
        """
        """
        for net, concns in self.known:
            self.assertEqual(mcautil.get_dynvarssvals(net).tolist(), concns)
    

class TestStoichMat(unittest.TestCase):
    known = [(net1, [[-1],[1]])]

    def test_known(self):
        """
        """
        for net, mat in self.known:
            self.assertEqual(mcautil.get_stoich_mat(net).tolist(), mat)
            

class TestLinkMat(unittest.TestCase):
    known = [(net1, )]


class TestRedStoichMat(unittest.TestCase):
    """
    
    """
    known = [(net1, )]
        
    
class TestParamElasMat(unittest.TestCase):
    known = [(net1, )]
    

class TestConcnElasMat(unittest.TestCase):
    known = [(net_path, [[-2/3],[2/3]])]
    
    def test_known(self):
        """
        """
        for net, mat in self.known:
            self.assertEqual(mcautil.get_concn_elas_mat(net, normed=True).tolist(), mat)
    

class TestConcnCtrlMat(unittest.TestCase):
    known = [(net1, )]
    
class TestConcnRespMat(unittest.TestCase):
    known = [(net1, )]

class TestFluxCtrlMat(unittest.TestCase):
    known = [(net1, )]
    

class TestFluxRespMat(unittest.TestCase):
    known = [(net1, )]
    


        

if __name__ == "__main__":
    unittest.main()
    