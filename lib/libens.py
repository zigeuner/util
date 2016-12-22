"""
Some utility functions for the SloppyCell ensemble 
constructions and analysis.
"""

from __future__ import division

try:
    from collections import OrderedDict as OD  # Python 2.7
except ImportError:
    import ordereddict as OD  # Python 2.6
from itertools import groupby
import copy
import cPickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import libnet
reload(libnet)
import libmca
reload(libmca)

from SloppyCell.ReactionNetworks import *


class MatrixEnsemble(np.ndarray):
    """
    testing.
    """
    
    def __new__(cls, mats, rowvarids=None, colvarids=None, costs=None, 
                energies=None):
        """
        costs: a measure of solely the goodness of the fit, without effects
               of priors 
        energies: both the goodness of the fit and distance to priors, if any
        """
        if hasattr(mats, 'rowvarids') and rowvarids is None:
            rowvarids = mats.rowvarids
        if hasattr(mats, 'colvarids') and colvarids is None:
            colvarids = mats.colvarids
        if hasattr(mats, 'costs') and costs is None:
            costs = mats.costs
        if hasattr(mats, 'energies') and energies is None:
            energies = mats.energies
        mats = np.array(mats)
        if mats.ndim == 2:
            mats = mats.reshape(mats.shape[0], 1, mats.shape[1])
        obj = np.asarray(mats).view(cls)
        obj.rowvarids = rowvarids
        obj.colvarids = colvarids
        obj.costs = np.array(costs)
        obj.energies = np.array(energies)
        return obj


    def __array__finalize__(self, obj):
        if obj is None: 
            return
        self.rowvarids = getattr(obj, 'rowvarids', None)
        self.colvarids = getattr(obj, 'colvarids', None)
        self.costs = getattr(obj, 'costs', None)
        self.energies = getattr(obj, 'energies', None)
        
    
    def slice(self, idxs=None, start=0, stop=None, step=1):
        if stop is None or stop > len(self):
            stop = len(self)
        if idxs is None:
            idxs = np.arange(start, stop, step, dtype='int')
        ens_slice = self[idxs]
        costs_slice = self.costs[idxs]
        energies_slice = self.energies[idxs]
        return MatrixEnsemble(ens_slice, rowvarids=self.rowvarids, 
                              colvarids=self.colvarids, 
                              costs=costs_slice, energies=energies_slice)

    
    def save(self, filepath):
        """
        """
        dat = (np.array(self), self.costs, self.energies, self.rowvarids,
               self.colvarids)
        fh = open(filepath, 'w')
        cPickle.dump(dat, fh)
        fh.close()
        
             
    @staticmethod
    def load(filepath):
        """
        """
        f = open(filepath)
        arr, costs, energies, rowvarids, colvarids = cPickle.load(f)
        f.close()
        ens = MatrixEnsemble(arr, rowvarids=rowvarids, colvarids=colvarids, 
                             costs=costs, energies=energies)
        return ens
    

    @property
    def matshape(self):
        return self[0].shape


    def copy(self):
        return copy.deepcopy(self)
    
    
    def matrix2array(self):
        """
        Works only for ensembles of 1-d matrix.
        """
        if self.matshape[0] != 1:
            raise StandardError, "matrix is not one-dimensional."
        return np.array([np.array(mat).flatten() for mat in self])
        

    def get_thinned_ensemble(self, b=None, k=None, random=False, N=None):
        """
        Remove burn-in portions, and decorrelate the sample.
        b: burn-in ratio
        k: correlation length
        """
        if random:
            idxs = np.random.permutation(len(self))[:N]
            try:
                costs = self.costs[idxs]
            except:
                costs = None
            try:
                energies = self.energies[idxs]
            except:
                energies = None
            ens = MatrixEnsemble(self[idxs], rowvarids=self.rowvarids,
                                 colvarids=self.colvarids, costs=costs,
                                 energies=energies)
            return ens
        else:
            return self.slice(start=int(len(self)*b), step=k)
        
    
    def get_unique_ensemble(self):
        """
        Return a new MatrixEnsemble instance with repeating elements removed.
        """
        # tolist because checking the uniqueness of ndarrays is ambiguous
        dat = [(self[i].tolist(), self.energies[i]) for i in range(len(self))]
        # remove duplicates; got the recipe from here:
        # http://stackoverflow.com/questions/10784390/python-eliminate-
        # duplicates-of-list-with-unhashable-elements-in-one-line
        dat = [k for k, v in groupby(sorted(dat))]
        ens = [np.matrix(item[0]) for item in dat]
        energies = [item[1] for item in dat]
        return MatrixEnsemble(ens, rowvarids=self.rowvarids,
                              colvarids=self.colvarids, energies=energies)
        
        
    def plot_trace(self, rowvarid='', colvarid='', filepath=''):
        """
        """
        if rowvarid:
            rowvaridx = self.rowvarids.index(rowvarid)
        else:
            rowvaridx = 0
        if colvarid:
            colvaridx = self.colvarids.index(colvarid)
        else:
            colvaridx = 0
        trace = self[:, rowvaridx, colvaridx]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(trace)
        ax.set_ylabel('value of '+rowvarid+colvarid)
        ax.set_xlabel('steps in the MCMC random walk')
        plt.savefig(filepath)
    
    
    def plot_traces(self, filepath=''):
        #fig = plot.figure(figsize=(20, 10), dpi=300)
        for i in range(len(self.colvarids)):
            pass
        
    
    def plot_acceptance_ratio_trace(self, windowsize=100, stepsize=50, 
                                    filepath=''):
        """
        """
        rs = []
        nsteps = np.arange(int(len(self)/stepsize), dtype='int')
        for i in nsteps:
            window = self.slice(start=i*stepsize, stop=i*stepsize+windowsize)
            rs.append(len(window.get_unique_ensemble())/len(window))
        fig = plt.figure(figsize=(10, 1.5), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(nsteps*stepsize, rs, '-', linewidth=0.5)
        #ax.set_title('Trace of Acceptance Ratio\
        #    (window size = %d, step size = %d)' % (windowsize, stepsize))
        #ax.set_ylabel('acceptance ratio')
        ax.set_ylim(0,1)
        #ax.set_xlabel('steps in the MCMC random walk')
        plt.subplots_adjust(top=0.94, bottom=0.12, left=0.05, right=0.95)
        plt.savefig(filepath, dpi=300)
        
    
    def get_sorted_ensemble(self):
        """
        Return a new MatrixEnsemble instance with elements sorted by energy.
        """
        dat = [(self[i], self.energies[i]) for i in range(len(self))]
        dat_sorted = sorted(dat, key=lambda item: item[1])
        return MatrixEnsemble(dat_sorted[0], rowvarids=self.rowvarids,
                              colvarids=self.colvarids, energies=dat_sorted[1])
        

    def plot_hist(self, idx):
        """
        idx: a multiple index (a tuple) in the case
        """
        pass


    def plot_checkerboard(self, log=True, initvals=None, figtitle='', 
                          filepath='', lims=None):
        """
        Pairwise correlation.
        Should not use for large matrices.
        lims: a list of lims
        """
        n = len(self.colvarids)
        fig = plt.figure(figsize=(n/2, n/2), dpi=300)
        for i, j in np.ndindex((n, n)):
            arr_i = self[:, 0, i]
            arr_j = self[:, 0, j]
            if log:
                arr_i = np.log(arr_i)
                arr_j = np.log(arr_j)
            varid_i = self.colvarids[i]
            varid_j = self.colvarids[j]
            if initvals is not None:
                initval_i = initvals[i]
                initval_j = initvals[j]
                if log:
                    initval_i = np.log(initval_i)
                    initval_j = np.log(initval_j)
            ax = plt.subplot(n, n, i*n+j+1)
            ax.scatter(arr_j, arr_i, s=0.4, marker='o', facecolor='k', lw = 0)
            ax.plot([initval_j], [initval_i], 'or', markersize=3.5)
            if lims is not None:
                lim_i = lims[i]
                lim_j = lims[j]
                ax.set_xlim(lim_j)
                ax.set_ylim(lim_i)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_xlabel(varid_j, fontsize=6)
                ax.xaxis.set_label_position('top')
            if i == n-1:
                ax.set_xlabel(varid_j, fontsize=6)
            if j == 0:
                ax.set_ylabel(varid_i, fontsize=6)
            if j == n-1:
                ax.set_ylabel(varid_i, fontsize=6)
                ax.yaxis.set_label_position('right')
        plt.subplots_adjust(wspace=0, hspace=0, top=0.95, bottom=0.05, 
                            left=0.05, right=0.95)
        plt.suptitle(figtitle, fontsize=20)
        plt.savefig(filepath, dpi=300)
        plt.close()
        

    def get_ground_ensemble(self, cost_cutoff=None, energy_cutoff=None):
        """
        """
        if cost_cutoff is not None:
            idxs = self.costs < cost_cutoff
        if energy_cutoff is not None:
            idxs = self.energies < energy_cutoff
        return self.slice(idxs=idxs)
    
    
    def plot_QQ_Boltz(self, ninterval=100, filepath='',
                    figtitle='Quantile-Quantile Plot of Energy Distribution'):
        """
        """
        ps = np.arange(0, 1., 1./ninterval)
        ## Get the quantiles of data (energies).
        energies = np.sort(self.energies)
        energy_max = energies[-1]
        energy_quantiles = sp.stats.mstats.mquantiles(energies, ps)
        ## Get the quantiles of Boltzmann distribution.
        probs = np.exp(-energies)/np.sum(np.exp(-energies))
        cumprobs = np.cumsum(probs)
        idxs = [np.abs(cumprobs-p).argmin() for p in ps]
        energy_quantiles_boltz = energies[idxs]
        ## Make the figure.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(energy_quantiles, energy_quantiles_boltz)
        ax.plot([0, energy_max], [0, energy_max], 'r')
        ax.set_aspect('equal')
        ax.set_xlim(0, energy_max)
        ax.set_ylim(0, energy_max)
        ax.set_xlabel('Energy of States in the Ensemble')
        ax.set_ylabel('Energy of States in Boltzmann Distribution')
        ax.set_title(figtitle)
        plt.savefig(filepath)


    def plot_hist_Boltz(self, bins=20, normed=True, xlim=None, figtitle='', 
                        filepath=''):
        """
        """
        fig = plt.figure(dpi=600)
        plt.hist(self.costs, bins=bins, normed=normed)
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        plt.xlim(xlim)
        plt.title(figtitle)
        plt.savefig(filepath)
        plt.close()
    
    
    def plot_linearization(self, J=None, JtJ=None, sing_val_cutoff=1e-5, 
                           dpi=300, filepath=''):
        """
        Log-log plot of parameter variation ranges, where log is log10, and 
        variation ranges are the distances between log10 of 0.025 and 0.975 
        quantiles.
        """
        if J is not None:
            JtJ = np.dot(J.transpose(), J)
        evals, evecs = np.linalg.eig(JtJ)
        V = np.matrix(evecs)
        D = np.matrix(np.diag(np.maximum(evals, evals[0] * sing_val_cutoff)))
        # times 4 because 4 sigmas are needed to get 0.95 coverage 
        # (2 sigmas in both directions)
        # divided by np.log(10) because JtJ is w.r.t. natural log of params
        ranges_linear = np.sqrt(np.diag(V * D.getI() * V.getT()))*4/np.log(10)
        quantiles = self.get_quantiles(ps=[0.025, 0.975], log=True)
        ranges_ensemble = [(quantiles[0,i,-1] - quantiles[0,i,0])
                           for i in range(len(evals))]
        ranges = list(ranges_linear) + ranges_ensemble
        min = np.int(np.floor(np.log10(np.min(ranges))))
        max = np.int(np.ceil(np.log10(np.max(ranges))))
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot([min, max], [min, max], 'r')
        ax.scatter(np.log10(ranges_linear), np.log10(ranges_ensemble))
        ax.set_xlabel('Linearization')
        ax.set_ylabel('Ensemble')
        ax.set_xticks(np.arange(min, max+0.1))  # +0.1 to include max
        ax.set_yticks(np.arange(min, max+0.1))
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        plt.title('Log-Log Scatterplot of Parameter Variation Ranges')
        plt.savefig(filepath, dpi=dpi)
        plt.close()
        
    
    def cmp_with_linearization(self, JtJ, log=True, energy_cutoff=None):
        """
        Return ...
        """
        # angle: a function that takes in two vectors and outputs 
        # an angle in degrees
        angle = lambda arr1, arr2: np.arccos(np.dot(arr1,arr2) /\
                (np.linalg.norm(arr1)*np.linalg.norm(arr2))) / np.pi * 180
        # note that numpy.linalg.eig outputs eigenvectors in the increasing
        # order of eigenvalues, thereby reversing the order of eigenvectors
        Lj, Vhj = np.linalg.svd(JtJ)[1:]
        if energy_cutoff:
            ens = self.get_low_energy_ensemble(energy_cutoff)
        else:
            ens = self.copy()
        if log:
            Le, Ve = Ensembles.PCA_eig_log_params(ens.matrix2array())
        else:
            Le, Ve = Ensembles.PCA_eig(ens.matrix2array())
        ratios = Le / Lj
        angles = [angle(Vhj.transpose()[i], Ve[i]) for i in range(len(Ve))]
        # a vector in the opposite direction of another vector is considered
        # to be in the same direction;
        # the following line does that correction
        angles = np.minimum(angles, 180-np.array(angles))
        return ratios, angles


    def get_quantiles(self, ps=[0.025,0.5,0.975], normalize=False, log10=False):
        """
        Compute the quantiles for the variation of each variable.
        """
        qs = np.ndarray(list(self.matshape) + [len(ps)])
        for i, j in np.ndindex(self.matshape):
            arr = self[:, i, j]
            qs[i, j] = sp.stats.mstats.mquantiles(arr, ps)

        if normalize:
            medians = np.median(self, axis=0)
            medians = np.repeat(medians, len(ps), axis=1).reshape(qs.shape)
            qs = qs / medians
        if log10:
            if np.any(qs < 0):
                raise ValueError("some quantiles are < 0 & can't take log.")
            else:
                qs = np.log10(qs)
        return qs


    def plot_quantiles(self, ps=[0.025,0.5,0.975], normalize=False, log10=False, 
                       initvals=None, sort_by_range=False, figtitle='', 
                       filepath=''):
        """
        """
        qs = self.get_quantiles(ps=ps, normalize=normalize, log10=log10)
        medians = np.median(self, axis=0)
        colvarids = self.colvarids
        
        if sort_by_range:
            ranges = qs[:,:,-1] - qs[:,:,0]    
        nrow, ncol = self.matshape
        fig = plt.figure(figsize=(ncol/2, 3), dpi=300)
        for i in range(nrow):
            ax = fig.add_subplot(nrow, 1, i)
            heights = qs[i,:,-1] - qs[i,:,0]
            bottoms = qs[i,:,0]
            ax.bar(np.arange(ncol)-0.4, heights, bottom=bottoms, width=0.8)
            if initvals is not None:
                if not isinstance(initvals, np.ndarray) or\
                    initvals.shape != self.matshape:
                    initvals = np.array(initvals).reshape(self.matshape)
                if normalize:
                    medians = np.median(self, axis=0)
                    initvals = initvals / medians
                if log10:
                    initvals = np.log10(initvals)
                ax.plot(np.arange(ncol), initvals[i,:], 'or', markersize=5)
            for j in range(ncol):
                for k in range(len(ps)):
                    q_var = qs[i, j, k]
                    ax.plot([j-0.4, j+0.4], [q_var, q_var], '-r', linewidth=1)
            if normalize:
                ax.plot([-1, ncol], [2,2], 'g--')
                ax.plot([-1, ncol], [-2,-2], 'g--')
            if self.rowvarids:
                ax.set_ylabel(self.rowvarids[i])
            ax.set_xlim(-1, ncol)
            if normalize:
                ax.set_ylim(-0.3,0.3)
            ax.yaxis.set_label_position('left')
            #ax2 = ax.twinx()
            #ax2.yaxis.set_label_position('right')
            ylabel = str(ps) + ' quantiles'
            if normalize:
                ylabel = ylabel + '\n normalized by medians'
            if log10:
                ylabel = '$log_{10}$(' + ylabel + ')'
            #ax.set_ylabel(ylabel, fontsize=10)
            #yticks = ax.get_yticks()
            yticks = np.arange(-0.25, 0.251, 0.125/2)
            ax.set_yticks(yticks)
            yticklabels = ['%.2g' % np.power(10, ytick) for ytick in yticks]
            ax.set_yticklabels(yticklabels, fontsize=10)
        if colvarids:
            xticklabels = colvarids
            if normalize:
                xticklabels = [colvarids[j] + '\n(' + '%.1E'%medians[i,j] +')' 
                               for j in range(len(colvarids))]
            plt.xticks(np.arange(ncol), xticklabels, 
                       rotation='vertical', size=10)
        plt.subplots_adjust(top=0.95, bottom=0.27, left=0.15, right=0.95)
        plt.title(figtitle)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
    
    def plot_hist_var(self, varid, log10=True, bins=20, refval=None, 
                      label_quantiles=True, filepath=''):
        """
        """
        ens_var = self[:, 0, self.colvarids.index(varid)]
        if log10:
            dat = np.log10(ens_var)
        else:
            dat = ens_var
              
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.hist(dat, bins=bins, normed=True)
        # label reference values (e.g., best fits)
        if refval:
            if log10:
                ax.plot([np.log10(refval)], [0], 'or', markersize=10)
            else:
                ax.plot([refval], [0], 'or', markersize=10)
        # label quantiles 
        if label_quantiles:
            qs = sp.stats.mstats.mquantiles(ens_var, [0.025, 0.25, 0.5, 0.75, 0.975])
            if log10:
                qs = np.log10(qs)
            for q in qs:
                ax.plot([q, q], [0, ax.get_ylim()[1]*0.05], '-r', markersize=5)
        # relabel xticks to the log10 scale
        if log10:
            xticks = ax.get_xticks()
            xticklabels = np.round(10**xticks, 2)  # keep 2 decimal places
            ax.set_xticklabels(xticklabels, rotation='vertical')

        ax.set_title('Distribution of %s in the Ensemble'%varid)
        plt.savefig(filepath, dpi=300)
        plt.close()


    def plot_traj_ensemble(self, net, T, varids=None, N=None, 
                           traj_timeseries=None, CV=None):
        """
        N: number of parameter sets to generate the traj ensemble
        """
        ens_unique = self.get_unique_ensemble()
        if N:
            ens_N = self.get_thinned_ensemble(random=True, N=N)
        else:
            ens_N = ens_unique
        if not varids:
            varids = traj_timeseries.key_column.keys()
        for varid in varids:
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            for paramvals in ens_N:
                paramvals = np.array(paramvals).flatten()
                net_copy = net.copy()
                net_copy.update_optimizable_vars(paramvals)
                traj = Dynamics.integrate(net_copy, [0, T])
                ax.plot(traj.timepoints, traj.get_var_traj(varid), '-b', 
                        linewidth=8, zorder=1)
            timeseries = traj_timeseries.get_var_traj(varid)
            ax.errorbar(traj_timeseries.timepoints, timeseries, 
                        yerr=timeseries*CV, fmt='or', zorder=10)
            ax.set_title(varid)
            ax.set_xlim(-T*0.1, T*1.1)
            plt.savefig('TrajEns_%s.png' % varid, dpi=300)
            plt.close()


def get_variable_ensemble(net, ens_paramvals, get_vars, rowvarids=None, 
                          colvarids=None, skip_fail=False, print_fail=False):
    """
    Return an ensemble instance
    get_vars: a function that takes a network and a set of parameter values 
              as the input and outputs variable values of interest
    skip:
    """
    dynvarvals_init = [var.value for var in net.dynamicVars]
    ens_vars = []
    count = 0
    for paramvals in ens_paramvals:
        paramvals = np.array(paramvals).flatten()
        paramvals = dict(zip(ens_paramvals.colvarids, paramvals))
        # convert the results to matrices as the function returns a 
        # MatrixEnsemble instance
        if skip_fail:
            try:
                vars = np.matrix(get_vars(net, paramvals))
            except ValueError:
                count = count + 1
                continue
        else:
            vars = np.matrix(get_vars(net, paramvals))
        # reset the dynamic variable values (for later steady state 
        # calculations)
        libnet.update_net(net, dynvarvals=dynvarvals_init)
        ens_vars.append(vars)
    ens_vars = MatrixEnsemble(ens_vars, rowvarids=rowvarids, 
                              colvarids=colvarids, costs=ens_paramvals.costs, 
                              energies=ens_paramvals.energies)
    if print_fail:
        print "Proportion of failed steady state evaluations: ",\
              count / len(ens_paramvals)
    return ens_vars



    """    
    ranges = [value[1]-value[0] for value in optvarid2info.values()]
    bottoms = [value[0] for value in optvarid2info.values()]
    medians = [value[2] for value in optvarid2info.values()]

    ## Get the initial values of parameters.
    if optvarvals0:
        if not hasattr(optvarvals0, 'items'):  # then not a mapping object
            optvarvals0 =  dict(zip(optvarids0, optvarvals0)) 
        # get the correct ordering.
        optvarvals0 = [optvarvals0.get(optvarid) for optvarid in optvarids]
        # optvarlnvals0: log-normalized initial values of optimizable variables
        optvarlnvals0 = np.log10(np.array(optvarvals0)/np.array(medians))

    if plot_medians:
        optvarids = [optvarids[i] + '\n(' + '%.1E' % medians[i] +')' 
                     for i in range(len(optvarids))]
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(optvarids))-0.4, ranges, bottom=bottoms, width=0.8)
    # plot the initial values of parameters
    if optvarvals0:
        ax.plot(np.arange(len(optvarids)), optvarlnvals0, 'or')
    ax.set_xticks(np.arange(len(optvarids)))
    ax.set_xticklabels(optvarids, size=7, rotation='vertical')
    ax.set_xlim(-0.5, len(optvarids)-0.5)
    ax.set_ylabel('$log_{10}$(normalized quantile)')
    ax.set_title(title_fig)
    #plt.subplots_adjust(left=0.25, right=0.7, bottom=0.2, top=0.8)
    plt.savefig(title_file)
    """ 



    """
    def get_lognormalized_quantiles(self, ps=[0.025,0.5,0.975], log=True, 
                                    sort_by_range=True):
        varid2ps = get_quantiles(ens, optvarids, ps=ps)
    
        ## Log-normalize the quantiles.
        log = eval('np.log'+str(logbase))
        # a function mapping from quantiles (qs) to 
        # log-normalized quantiles (lnqs)
        qs2lnqs = lambda ps: (log(qs[0]/qs[1]), 0, log(qs[2]/qs[1]))
        varid2lnqs = OD(zip(varid2qs.keys(), 
                            map(qs2lnqs, varid2qs.values())))

        ## Sort varid2lnqs by the range of variation.
        # range = log-normalized(qhigh) - log-normalized(qlow), 
        # minus b/c the latter is always negative
        if sort:
            varid2lnqs = OD(sorted(varid2lnqs.items(), 
                                   key=lambda item: item[1][2]-item[1][0]))
    
        ## Replace the log-normalized(median) (which is always 1) by 
        # the bare median.
        # a function mapping from old items (varid, (ln(plow), 1, ln(phigh))) to
        # new items (varid, (ln(qlow), ln(qhigh), median));
        # I call the tuple (ln(qlow), lh(qhigh), median) "info"
            olditem2newitem = lambda item: (item[0], 
                                (item[1][0], item[1][2], varid2qs[item[0]][1]))
            varid2info = OD(map(olditem2newitem, varid2lnqs.items()))
            return varid2info
    """


"""
    optvarids = optvarid2info.keys()
    ranges = [value[1]-value[0] for value in optvarid2info.values()]
    bottoms = [value[0] for value in optvarid2info.values()]
    medians = [value[2] for value in optvarid2info.values()]

    ## Get the initial values of parameters.
    if optvarvals0:
        if not hasattr(optvarvals0, 'items'):  # then not a mapping object
            optvarvals0 =  dict(zip(optvarids0, optvarvals0)) 
        # get the correct ordering.
        optvarvals0 = [optvarvals0.get(optvarid) for optvarid in optvarids]
        # optvarlnvals0: log-normalized initial values of optimizable variables
        optvarlnvals0 = np.log10(np.array(optvarvals0)/np.array(medians))

    if plot_medians:
        optvarids = [optvarids[i] + '\n(' + '%.1E' % medians[i] +')' 
                     for i in range(len(optvarids))]
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(optvarids))-0.4, ranges, bottom=bottoms, width=0.8)
    # plot the initial values of parameters
    if optvarvals0:
        ax.plot(np.arange(len(optvarids)), optvarlnvals0, 'or')
    ax.set_xticks(np.arange(len(optvarids)))
    ax.set_xticklabels(optvarids, size=7, rotation='vertical')
    ax.set_xlim(-0.5, len(optvarids)-0.5)
    ax.set_ylabel('$log_{10}$(normalized percentile)')
    ax.set_title(title_fig)
    #plt.subplots_adjust(left=0.25, right=0.7, bottom=0.2, top=0.8)
    plt.savefig(title_file)
"""
