"""
Some utility functions for the SloppyCell sampling 
constructions and analysis.
"""

from __future__ import division

from collections import OrderedDict as OD
from itertools import groupby
import copy
import cPickle

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot

from SloppyCell.ReactionNetworks import *

from util2 import butil, plotuti
reload(butil)


class MCAMatrix(np.matrix):
    """
    stoichiometry matrix,
    elasticity matrix of concentrations,
    elasticity matrix of parameters,
    concentration control matrix 
    flux control matrix
    concentration response matrix
    flux response matrix

    reduced stoichiometry matrix
    link matrix L
    L0    
    """
    def __new__(cls, mat, rowvarids=None, colvarids=None, dtype='float'):
        obj = np.matrix(mat, dtype=dtype).view(cls)
        obj.rowvarids = rowvarids
        obj.colvarids = colvarids
        return obj

 
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.rowvarids = getattr(obj, 'rowvarids', None)
        self.colvarids = getattr(obj, 'colvarids', None)

    
    def get_row(self, rowvarid):
        i = self.rowvarids.index(rowvarid)
        return self[i]
        
        
    def get_element(self, rowvarid, colvarid):
        i = self.rowvarids.index(rowvarid)
        j = self.colvarids.index(colvarid)
        return self[i, j]
    
    
    def to_df(self):
        return pd.DataFrame(self, index=self.rowvarids, columns=self.colvarids)
    
    
    def vstack(self, other):
        """
        Stack vertically.
        """
        mat = MCAMatrix(np.vstack((self, other)), 
                        rowvarids=self.rowvarids+other.rowvarids,
                        colvarids=self.colvarids)
        return mat


def get_stoich_mat(net=None, rxnid2stoich=None, dynamic=True):
    """
    Return the stoichiometry matrix (N) of the given network or 
        dict rxnid2stoich.
    Rows correspond to species (net.dynamicVars);
    columns correspond to reactions (net.reactions).
    Input:
        net & rxnid2stoich: one and only one of them should
                            be given
        dynamic: if True, use net.dynamicVars.keys() as row ids
                 if False, use net.species.keys() as row ids
    """
    if net:
        attr = 'stoich_mat'
        try:
            return get_matrix_trial(net, attr)
        except AttributeError:
            if dynamic:
                varids = net.dynamicVars.keys()
            else:
                varids = net.species.keys()
            rxnids = net.reactions.keys()
            mat = MCAMatrix(np.zeros((len(varids), len(rxnids))),
                            rowvarids=varids, colvarids=rxnids)
            for i in range(len(varids)):
                for j in range(len(rxnids)):
                    varid = varids[i]
                    rxnid = rxnids[j]
                    try:
                        stoichcoef = net.reactions.get(rxnid).\
                                     stoichiometry[varid]
                        if isinstance(stoichcoef, str):
                            stoichcoef = net.evaluate_expr(stoichcoef)
                        mat[i, j] = stoichcoef
                    except KeyError:
                        pass  # mat[i,j] remains zero
                    
            for i in range(mat.shape[1]):
                col = mat[:,i]
                nums = [num for num in col.flatten() if num]
                denoms = [fractions.Fraction(num).denominator for num in nums]
                denom = np.prod(list(set(denoms)))
                mat[:,i] = col * denom
                        
            setattr(net, attr, mat)
            setattr(net, short2long(attr), mat)
        
            # update the meta-attribute 'attrs_structural'
            if not hasattr(net, 'attrs_structural'):
                net.attrs_structural = set([])
            net.attrs_structural.update([attr, short2long(attr)])
    
    if rxnid2stoich:
        rxnids = rxnid2stoich.keys()
        varids = [stoich.keys() for stoich in rxnid2stoich.values()]
        varids = list(set(butil.flatten(varids)))
        mat = MCAMatrix(np.zeros((len(varids), len(rxnids))),
                        rowvarids=varids, colvarids=rxnids)
        for i in range(len(varids)):
            for j in range(len(rxnids)):
                varid = varids[i]
                rxnid = rxnids[j]
                try:
                    stoichcoef = rxnid2stoich[rxnid][varid]
                    mat[i, j] = stoichcoef
                except KeyError:
                    pass  # mat[i,j] remains zero
                                
    return mat
get_stoichiometry_matrix = get_stoich_mat


def get_pool_mul_mat(net):
    """
    Return a matrix whose row vectors are multiplicities of dynamic variables
    in conservation pools. 
    Mathematically, the matrix has rows spanning the left null space of the
    stoichiometry matrix of the network.
    
    The function is computationally costly, because it calls *sage* to perform 
    matrix computations over the integer ring. 
    (Note that the matrix is converted to floats before being returned.)
    """
    attr = 'pool_mul_mat'
    
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        ## The following codes compute the INTEGER basis of left null space
        #  of stoichiometry matrix.

        ## Convert the matrix into a string recognizable by sage.
        stoichmat = get_stoich_mat(net)
        matstr = re.sub('\s|[a-z]|\(|\)', '', stoichmat.__repr__())

        ## Write a (sage) python script "tmp_sage.py".
        # for more info of the sage commands: 
        # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
        # -import-sage-into-a-python-script
        # http://www.sagemath.org/doc/tutorial/tour_linalg.html
        f = open('.tmp_sage.py', 'w')
        f.write('from sage.all import *\n\n')
        f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integers as the field
        f.write('print A.kernel()')  # this returns the left nullspace vectors
        f.close()

        ## Call sage and run mat.py.
        out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                               stdout=subprocess.PIPE)
        
        ## Process the output from sage.
        vecstrs = out.communicate()[0].split('\n')[2:-1]
        vecs = [eval(re.sub('(?<=\d)\s*(?=\d|-)', ',', vec)) 
                for vec in vecstrs]
        
        mat = MCAMatrix(vecs, colvarids=net.dynamicVars.keys())
        setattr(net, attr, mat)
        setattr(net, short2long(attr), mat)
        
        ## Update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_pool_multiplicity_matrix = get_pool_mul_mat


def get_ss_flux_mat(net=None, stoichmat=None):
    """
    Input:
        net & stoichmat: one and only one of them should be given
    """
    attr = 'ss_flux_mat'
    try:
        return get_matrix_trial(net, attr)
    except AttributeError:
        ## The following codes compute the INTEGER basis of right null space
        ## of stoichiometry matrix.

        ## convert the matrix into a string recognizable by sage
        if net:
            stoichmat = get_stoich_mat(net)
        matstr = re.sub('\s|[a-z]|\(|\)', '', stoichmat.__repr__())

        ## write a (sage) python script ".tmp_sage.py"
        # for more info of the sage commands: 
        # http://www.sagemath.org/doc/faq/faq-usage.html#how-do-i
        # -import-sage-into-a-python-script
        # http://www.sagemath.org/doc/tutorial/tour_linalg.html
        
        f = open('.tmp_sage.py', 'w')
        f.write('from sage.all import *\n\n')
        f.write('A = matrix(ZZ, %s)\n\n' % matstr)  # integers as the field
        f.write('print kernel(A.transpose())')  # return right nullspace vectors
        f.close()
        
        ## call sage and run .tmp_sage.py
        out = subprocess.Popen(['sage', '-python', '.tmp_sage.py'],
                               stdout=subprocess.PIPE)
        
        ## process the output from sage
        vecstrs = out.communicate()[0].split('\n')[2:-1]
        #vecs = [eval(re.sub('(?<=\d)\s*(?=\d|-)', ',', vec)) 
        #        for vec in vecstrs]
        vecs = [vec.strip('[]').split(' ') for vec in vecstrs]
        vecs = [[int(elem) for elem in vec if elem] for vec in vecs]
        mat = MCAMatrix(np.transpose(vecs), colvarids=stoichmat.colvarids)
        
        if net:
            setattr(net, attr, mat)
            setattr(net, short2long(attr), mat)
            # update the meta-attribute 'attrs_structural'
            if not hasattr(net, 'attrs_structural'):
                net.attrs_structural = set([])
            net.attrs_structural.update([attr, short2long(attr)])
        
        return mat
get_steadystate_flux_matrix = get_ss_flux_mat


def get_pools(net):
    """
    Return the conservation pools of the network. 
    pools: ordereddicts mapping from dynamic variable ids to
           their multiplicities in the pool.
    """
    attr = 'pools'
    
    try:
        return getattr(net, attr)
    except AttributeError:
        poolmat = get_pool_mul_mat(net)
        vecs = poolmat.tolist()
        pools = [OD(zip(net.dynamicVars.keys(), vec)) for vec in vecs]
        setattr(net, attr, pools)
        
        # update the meta-attribute 'attrs_structural'
        if not hasattr(net, 'attrs_structural'):
            net.attrs_structural = set([])
        net.attrs_structural.add(attr)
        
        return pools