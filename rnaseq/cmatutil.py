"""
"""

from __future__ import division
import copy
import cPickle
import time

import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

import plot
reload(plot)


class CorrMat(np.matrix):
    def __new__(cls, mat, geneids=None, dtype='float32'):
        obj = np.matrix(mat, dtype=dtype).view(cls)
        obj.geneids = geneids
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.geneids = getattr(obj, 'geneids', None)


    def save(self, filepath=''):
        dat = (np.matrix(self), self.geneids)
        fh = open(filepath, 'w')
        cPickle.dump(dat, fh)
        fh.close()


    @staticmethod
    def load(filepath=''):
        fh = open(filepath)
        mat, geneids = cPickle.load(fh)
        fh.close()
        cmat = CorrMat(mat, geneids)
        return cmat


    @staticmethod
    def construct_from_data(data, nan2zero=True, diag2zero=True, thrsh=None,
                            filepath='', print_time=True, print_header=''):
        """
        Input: 
            data: an instance of rnasequtil.Data
        """
        dmat = np.array(data, dtype='float32')
        t1 = time.time()
        cmat = np.corrcoef(dmat)
        t2 = time.time()
        if print_time:
            print str(print_header), t2 - t1
        cmat = np.array(cmat, dtype='float32')
        if nan2zero:
            cmat = np.nan_to_num(cmat)
        if diag2zero:
            np.fill_diagonal(cmat, 0)
        if thrsh:
            cmat = cmat.thresholding(thrsh)
        cmat = CorrMat(cmat, geneids=data.geneids)
        if filepath:
            cmat.save(filepath)
        return cmat


    def thresholding(self, thrsh):
        cmat = copy.deepcopy(self)
        cmat[np.abs(cmat) < thrsh] = 0
        return cmat


    def sparsify(self, format='coo', filepath=''):
        """
        Input: 
            format: 'coo', 'lil', etc.
        """
        csmat = getattr(sparse, format+'_matrix')(self)
        if filepath:
            dat = (csmat, self.geneids)
            fh = open(filepath, 'w')
            cPickle.dump(dat, fh)
            fh.close()
        csmat.geneids = self.geneids
        return csmat

    
    def get_density(self):
        num_nonzero = len(np.nonzero(self)[0])
        return num_nonzero / self.shape[0]**2


    @staticmethod
    def load_corrmat_sparse(filepath):
        fh = open(filepath)
        csmat, geneids = cPickle.load(fh)
        fh.close()
        csmat.geneids = geneids
        return csmat


    def flatten(self):
        idxs = np.triu_indices(self.shape[0], 1)
        ccs = self[idxs]
        return ccs

    
    def simplify(self):
        """
        Remove singletons.
        """
        pass



def get_eigvals(csmat, k, tol=1e-6, print_time=True, print_header=''):
    """
    Work for both regular and sparse correlation matrices.
    Return sorted eigenvalues.
    """
    t1 = time.time()
    eigvals = eigsh(csmat, k, tol=tol, return_eigenvectors=False)
    t2 = time.time()
    if print_time:
        print str(print_header), t2 - t1
    return np.sort(eigvals)


"""
cmat_rice = get_cmat(data_rice, nan2zero=True, diag2zero=True, 
                     filepath='Data/processed/dat_corrcoefmat_rice', 
                     print_time=True)
cmat_maize = get_cmat(data_maize, nan2zero=True, diag2zero=True, 
                      filepath='Data/processed/dat_corrcoefmat_maize', 
                      print_time=True)
cmat_maize_uniq = get_cmat(data_maize_uniq, nan2zero=True, diag2zero=True, 
                      filepath='Data/processed/dat_corrcoefmat_maize_uniq',
                      print_time=True)
"""

#ccs_maize = get_ccs(cmat_maize)
#plotutil.plot_hist(ccs_maize, bins=np.linspace(-1,1,201), xlabel='corr coef', 
#                  filepath='plot_hist_corrcoefs_maizeleaf_all.png')

