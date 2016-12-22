"""
ABBREVIATIONS AND TERMINOLOGIES:
var, variable:
idx, index:
val, value:
molwt, molecular weight:
wt, weight:
gn, genome:
pn, phenome:
gt, genotype:
pt, phenotype:
params, parameters:
ind, individual:
pop, population:
fit, fitness:
prnt, parent:
arg, argument:
kwarg, keyword argment:
fh, filehandle:
arr, array:
cp, copy:
mod, model:
dat, data:
fig, figure:

pth, path:
spth/sdir, save path/directory: path/directory for saving (dat/fig, etc.) files
rpth/rdir, read path/directory: path/directory for reading (dat/fig, etc.) files
"""

from __future__ import division
import random  # used in Population.recombine
import pickle  # used in Lineages.pickle and Lineages.load
import os      # used in Lineages
import copy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *


## check the net (need to copy? steady states dependent on the history? nonconstantVars updated?)

## steady state checkings
## beta=0 case
## seed the randnum generators


class Genome(np.ndarray):
    def __new__(cls, arrlike=None, vs=None, molwts=None, wt0=None, net=None):
        if net:
            arrlike = np.array([net.variables.get(id).value for id in net.gtids])
            vs = net.gtvs
            molwts = np.array([net.variables.get(id).molwt for id in net.gtids])
            wt0 = np.dot(arrlike, molwts)
        gn = np.array(arrlike).view(cls)
        gn.vs = vs
        gn.molwts = molwts
        gn.wt0 = wt0
        return gn
        
    def get(self, v):
        idx = self.vs.index(v)
        val = self[idx]
        return val
    
    def get_keyedlist(self):
        """
        return a keyedlist representation
        """
        return KeyedList(zip(self.vs, self))
        
    def leap(self, CV=0.05):
        """
        """
        arr2 = self + np.random.randn(len(self)) * (self * CV)  # sigma = self * CV
        while (arr2 < 0).any():
            arr2 = self + np.random.randn(len(self)) * (self * CV)
        gn2 = Genome(arrlike=arr2)
        gn2.__dict__ = self.__dict__
        return gn2
    
    def get_wt(self):
        wt = np.dot(self, self.molwts)
        return wt
    
    def normalize(self, r=1):
        """
        in-place normalize gn so that wt remains controlled
        """
        wt0, wt = self.wt0, self.get_wt()
        ratio = wt0 / wt * r
        self *= ratio
        
    def mutate(self, params):
        CV, normalize, r = params['CV'], params['normalize'], params['r']
        gn2 = self.leap(CV=CV)
        if normalize:
            gn2.normalize(r=r)
        return gn2
    
    def get_updated_net(self, net, cp=True):
        """
        cp: deepcopy the net, which takes ~0.1 second
        """
        if cp:
            net2 = net.copy()
        else:
            net2 = net
        for v in self.vs:
            val = self.get(v)
            net2.set_var_ic(net2.v2id[v], val)  # set_var_val not working
        return net2
        
    def get_phenome(self, net, cp=True, T=1000, derivs=False):
        net2 = self.get_updated_net(net=net, cp=cp)
        traj = Dynamics.integrate(net2, [0, T], return_derivs=derivs)
        id2vss = dict(zip(traj.key_column.keys(), np.transpose(traj.values)[:,-1]))
        pn = Phenome(arrlike=np.array(my.get_vals(id2vss, net2.ptids)), vs=net2.ptvs)
        if derivs:
            ks = [(ptid, 'time') for ptid in net2.ptids]
            dsss = np.array(my.get_vals(id2vss, ks))  ## not normalized
            return pn, dsss
        else:
            return pn
        
    def scatterplot(self, gn, log10=True):
        """
        The method produces a scatterplot of self and input gn for comparison.
        """
        xs, ys = self, gn
        fig = plt.figure(figsize=(13, 7), dpi=130)
        ax = fig.add_subplot(111)
        if log10:
            xs, ys = np.log10(xs), np.log10(ys)
        ax.plot(xs, ys, 'o')
        # plot the diagonal line for reference
        xy_min = np.array([xs, ys]).min()
        xy_max = np.array([xs, ys]).max()
        ax.plot([xy_min, xy_max], [xy_min, xy_max], c='r')
        if log10:
            ax.set_xlabel('log10(initial genome)')
            ax.set_ylabel('log10(evolved genome)')
        else:
            ax.set_xlabel('initial genome')
            ax.set_ylabel('evolved genome')
        ax.set_xlim(xy_min, xy_max)  ## it does not work, why?
        ax.set_ylim(xy_min, xy_max)
        ax.axis('equal')
        ax.set_title('scatterplot of two genomes for comparison')
        plt.show()
        
    def get_dist(self, gn, gn0=None, mode='median'):
        """
        mode: median, mean, norm (L1, L2, Linf)
        gn0: scaled? ...
        """
        diff = np.abs(gn - self)
        if gn0:
            diff = diff / gn0
        if mode == 'median':
            return np.median(diff)
        elif mode == 'mean':
            return np.mean(diff)
        else
            return np.linalg.norm(diff, mode)
        
class Phenome(np.ndarray):
    def __new__(cls, arrlike=None, vs=None, net=None):
        if net:
            arrlike = [net.variables.get(ptid).value for ptid in net.ptids]
            vs = net.ptvs
        pn = np.array(arrlike).view(cls)
        pn.vs = vs
        return pn
        
    def get(self, v):
        idx = self.vs.index(v)
        val = self[idx]
        return val
    
    def get_keyedlist(self):
        """
        return a keyedlist representation
        """
        return KeyedList(zip(self.vs, self))
    
    def scatterplot(self, pn, log10=True):
        """
        The method produces a scatterplot of self and input pn for comparison.
        """
        xs, ys = self, pn
        fig = plt.figure(figsize=(13, 7), dpi=130)
        ax = fig.add_subplot(111)
        if log10:
            xs, ys = np.log10(xs), np.log10(ys)
        ax.plot(xs, ys, 'o')
        # plot the diagonal line for reference
        xy_min = np.array([xs, ys]).min()
        xy_max = np.array([xs, ys]).max()
        ax.plot([xy_min, xy_max], [xy_min, xy_max], c='r')
        if log10:
            ax.set_xlabel('log10(initial phenome)')
            ax.set_ylabel('log10(evolved phenome)')
        else:
            ax.set_xlabel('initial phenome')
            ax.set_ylabel('evolved phenome')
        ax.set_xlim(xy_min, xy_max)  ## it does not work, why?
        ax.set_ylim(xy_min, xy_max)
        ax.axis('equal')
        ax.set_title('scatterplot of two phenomes for comparison')
        plt.show()
        
class Parameters(dict):
    def __init__(self, **kwargs):
        for kw, arg in kwargs.items():
            self[kw] = arg
        
    def mutate(self):
        return self
        
class Individual(object):
    def __init__(self, gn=None, pn=None, derivs=None, params=None):
        self.gn = gn
        self.pn = pn
        self.params = params
        self.derivs = derivs
        
class Population(list):
    def get_subpop(self, idxs):
        #subpop = Population(my.get_sub(self, idxs))
        return my.get_sub(self, idxs)

    def marriage(self, plan):
        parents = self.get_subpop(plan)
        return parents
        
    def recombine_params(self):
        if len(self) == 1:
            params = self[0].params
        else:
            pass
        return params
    
    def recombine_gn(self):
        if len(self) == 1:
            gn = self[0].gn
        else:
            pass
        return gn
        
    def reproductor(self, plans):
        """
        Beyer & Schwefel 2002, line 5-13 (without line 11, fitness evaluation);
        gn is mutated, out of sync with pn (set to None in the new ind)
        """
        pop2 = Population()
        for plan in plans:
            parents = self.marriage(plan)
            params = parents.recombine_params()
            gn = parents.recombine_gn()
            params_tilde = params.mutate()
            gn_tilde = gn.mutate(params_tilde)
            ind = Individual(gn=gn_tilde, params=params_tilde)
            pop2.append(ind)
        return pop2
        
    def update_phenome(self, net, T=1e3, cp=True):
        """
        in place sync pn with gn
        """
        for ind in self:
            ind.pn, ind.derivs = ind.gn.get_phenome(net=net, T=T, cp=cp, derivs=True)
    
    def get_phenotypes(self, ptv='A_Cel', deriv0=1e-6, warn=True):
        """
        pts: an array of pt corresponding to ptv
        """
        if deriv0:
            pts = []
            for ind in self:
                if np.abs(ind.derivs).max() > deriv0:
                    pts.append(0)
                    if warn:
                        print "one individual discarded due to failure to reach ss"
                else:
                    pts.append(ind.pn.get(ptv))
        else:
            pts = [ind.pn.get(ptv) for ind in self]
        return np.array(pts)
    
    @staticmethod    
    def pts2fits(pts, beta=10, EO=False):
        """
        fit (fitness) of an ind represents its relative contribution 
        to the next generation (sum(fits) == 1). 
        If sampling is done deterministically, fit is the relative proportion;
        if stochastically, fit is the sampling probability.
        
        beta: a parameter for scaling
        if beta > 1, individual variation is increased; decreased if < 1.
        
        EO: equal opportunity; equal fits if True.
        
        pts_sc: scaled pts
        pts_sc_nml: normalized scaled pts
        fits: an array of fit
        """
        if EO:
            fits = np.ones((len(pts))) / len(pts)
        else:
            pts_sc = pts ** beta
            pts_sc_nml = pts_sc / pts_sc.sum()
            fits = pts_sc_nml
        return fits
        
    @staticmethod
    def fits2nchns(fits, lamb, stochastic=True):
        if stochastic:  # stochastic sampling
            cumsums = np.cumsum(fits)
            randnums = np.random.uniform(0, cumsums[-1], lamb)
            nchns = np.histogram(randnums, np.insert(cumsums, 0, 0))[0]
        else:  # deterministic sampling
            nchns = np.array(np.round(fits * lamb), dtype='int64')
            delta = lamb - nchns.sum()  # delta may not be 0 due to rounding errors 
            if delta != 0:  
                # change max to make nchns.sum() == lamb
                idx = my.get_indexes(nchns, maximum=True, k=1)[0]
                nchns[idx] += delta
        return nchns
        
    def truncate(self, mu, ptv='A_Cel', beta=10):
        fits = self.get_fitness(ptv=ptv, beta=beta, EO=False)
        idxs = my.get_indexes(fits, maximum=True, k=mu)
        subpop = self.get_subpop(idxs)
        return subpop
    
    def get_parents(self, nchns):
        """
        """
        idxs_nchnNonzero = (nchns != 0)
        pop_prnt = Population(np.array(self)[idxs_nchnNonzero])
        nchns_nonzero = nchns[idxs_nchnNonzero]
        plans = list()
        for idx_prnt in range(len(pop_prnt)):
            nchn = nchns_nonzero[idx_prnt]
            plans.extend([[idx_prnt]] * nchn)
        return pop_prnt, plans
        
    def recombine(self, rho=2):
        """
        """
        plans = list()
        for i in range(lamb):
            plans.append(random.sample(range(len(self)), rho))
        return plans
    
    def selector(self, lamb, comma=True, pop_prnt0=None, rho=1, truncation=False, 
                 mu=None, ptv='A_Cel', beta=10, EO=False, stochastic=True, deriv0=1e-6, warn=True):
        """
        scheme: ...
        """
        if comma:
            pop_off0 = self
        else:
            pop_off0 = self + pop_prnt0        
        if rho == 1:  # no recombination
            if truncation:
                pop_off0 = pop_off0.truncate(mu=mu, ptv=ptv, beta=beta)
            pts = pop_off0.get_phenotypes(ptv=ptv, deriv0=deriv0, warn=warn)
            fits = Population.pts2fits(pts=pts, beta=beta, EO=EO)
            nchns = Population.fits2nchns(fits=fits, lamb=lamb, stochastic=stochastic)
            pop_prnt1, plans = pop_off0.get_parents(nchns)
        else:  # recombination
            pop_prnt1 = pop_off0.truncate(mu=mu, ptv=ptv, beta=beta)
            plans = pop_prnt1.recombine(rho=rho)
        return pop_prnt1, plans
        
    def get_genomes(self):
        gns = np.array([ind.gn for ind in self])
        return gns
    gns = property(fget=get_genomes)
    
    def get_phenomes(self):
        pns = np.array([ind.pn for ind in self])
        return pns        
    pns = property(fget=get_phenomes)
    
    def plot_stats(self, ptv='A_Cel', beta=10, lamb=20):
        """
        """
        pts = self.get_phenotypes(ptv=ptv)
        fits = Population.pts2fits(pts, beta=beta, EO=False)
        nchns = np.array(Population.fits2nchns(fits, lamb=lamb, stochastic=True),
                         dtype='float')
        pts /= pts.sum()
        fits /= fits.sum()
        nchns /= nchns.sum()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs = np.arange(1, len(fits)+1)
        width = 0.2
        bars1 = ax.bar(xs-width, pts, width, color='r')
        bars2 = ax.bar(xs, fits, width, color='g')
        bars3 = ax.bar(xs+width, nchns, width, color='b')
        ax.set_ylim(0, 1)
        ax.set_ylabel('relative scale', fontsize=20)
        ax.set_xlim(0.5, len(fits)+0.5)
        ax.set_xticks(range(1, len(self)+1))
        ax.legend((bars1[0], bars2[0], bars3[0]), ('pt', 'fit', 'nchn'))
        ax.set_title('$\\beta$ = '+str(beta), fontsize=22)
        plt.show()
    
class Lineage(np.ndarray):
    def __new__(cls, arrlike=None):
        if arrlike == None:
            arrlike = []
        lin = np.array(arrlike).view(cls)
        return lin
        
    def append(self, pop):
        lin = list(self)
        lin.append(pop)
        lin = Lineage(arrlike=lin)
        return lin
        
class Lineages(np.ndarray):
    def __new__(cls, arrlike=None):
        """
        """
        if arrlike == None:
            arrlike = []
        lins = np.array(arrlike).view(cls)
        return lins
        
    def append(self, lin):
        lins = list(self)
        lins.append(lin)
        lins = Lineages(arrlike=lins)
        return lins
        
    def pickle(self, spth=''):
        """
        """
        if os.path.exists(spth):
            raise StandardError('file %s already exists' %spth)
        sdir = pth2dir(spth)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        fh = open(spth, 'w')
        pickle.dump(self, fh)
        fh.close()
        
    @staticmethod
    def load(rpth, recast=False):
        """
        """
        fh = open(rpth)
        lins = pickle.load(fh)
        if recast:
            lins = Lineages(lins)
        fh.close()
        return lins
        
    def make_genome(self, l=1, g=1, n=1, gn0=None, net=None):
        """
        """
        arr = self[l-1, g-1, n-1]
        gn = Genome(arrlike=arr)
        if net:
            gn0 = Genome(net=net)
        if gn0:
            gn.__dict__ = gn0.__dict__
        return gn
    
    def make_phenome(self, l=1, g=1, n=1, pn0=None, net=None):
        """
        """
        arr = self[l-1, g-1, n-1]
        pn = Phenome(arrlike=arr)
        if net:
            pn0 = Phenome(net=net)
        if pn0:
            pn.__dict__ = pn0.__dict__
        return pn
    
    def plot_evo(self, ome0, vs_plot=None, step=1, show_average=True, save=True, sdir='', show=False):
        """
        ome0: gn0 (genome0) or pn0 (phenome0)
        vs_plot: the vs for plotting
        step: the generation step at which values are plotted
        """
        shape = self.shape
        L, G, N = shape[0], shape[1], shape[2]
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        if vs_plot:
            vs = vs_plot
        else:
            vs = ome0.vs

        for i in range(len(vs)):
            v = vs[i]
            vss0 = ome0[i]
            fig = plt.figure(figsize=(13, 7), dpi=130)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(0, 1, 'or', ms=6)
            for l in range(L):
                width = step * 0.4
                if L == 1:
                    offset = 0
                else:
                    offset = width * (l / (L - 1) - 0.5)
                for g in range(0, G, step):
                    vss_pop = self[l, g, :, i] / vss0
                    g_pop = np.zeros(vss_pop.shape) + (g + 1) + offset
                    ax.plot(g_pop, vss_pop, 'o', c='brcmygk'[l%7], ms=1, mec='brcmygk'[l%7])
                if show_average:
                    ax.plot(range(0, G+1), np.insert(self[l, :, :, i].mean(1)/vss0, 0, 1), 
                            c='brcmygk'[l%7], lw=2)
            ax.set_xlim(-1, G+1)
            ax.set_xlabel('Generation', fontsize=14)
            ax.set_title(v+' ('+str(round(vss0, 3))+')', fontsize=16)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
            if save:
                spth = sdir+v+'.png'
                if os.path.exists(spth):
                    raise StandardError('file %s already exists' %spth)
                plt.savefig(spth, dpi=130)
            if show:
                plt.show()
            else:
                plt.close()  # if not close, figs pop up next time upon calling show()
    
    @staticmethod
    def snapshot2Fst(snapshot):
        within = np.mean(snapshot.std(1))
        means = snapshot.mean(1)
        dists = list()
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                dists.append(abs(means[i] - means[j]))
        between = np.mean(dists)
        Fst = between / (within + between)
        return Fst
    
    @staticmethod
    def snapshot2Fst2(snapshot):
        within = 
        
    def plot_Fst(self, ome0, nv_subplot=7, save=True, spth='', show=False):
        """
        """
        shape = self.shape
        L, G, N = shape[0], shape[1], shape[2]
                
        vs = ome0.vs
        vs_cp = copy.copy(vs)
        lins_cp = self.copy()        
        nsubplot = int(np.ceil(len(vs_cp) / nv_subplot))
        plt.figure(figsize=(15, nsubplot*2.5), dpi=250)
        for i in range(nsubplot):
            plt.subplot(nsubplot, 1, i+1)
            nv = min(nv_subplot, len(vs_cp))
            vs_subplot = vs_cp[0:nv]
            lins_subplot = lins_cp[:, :, :, 0:nv]
            
            for j in range(nv):
                Fsts = map(lambda snapshot: Lineages.snapshot2Fst(snapshot), 
                           lins_subplot[:, :, :, j].swapaxes(0, 1))
                plt.plot(range(1, G+1), Fsts)
                if vs_cp:
                    vs_cp = vs_cp[1:]
                    lins_cp = lins_cp[:, :, :, 1:]
                
            plt.legend(vs_subplot, loc='upper left', 
                       bbox_to_anchor = (0.05, 1), prop={'size': 7})
            plt.ylim(0, 1)
            plt.ylabel('Fst', fontsize=14, rotation='horizontal')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        if save:
            if os.path.exists(spth):
                raise StandardError('file %s already exists' %spth)
            sdir = pth2dir(spth)
            if not os.path.exists(sdir):
                os.makedirs(sdir)     
            plt.savefig(spth, dpi=130)
        if show:
            plt.show()
        else:
            plt.close()  # if not close, figs pop up next time upon calling show()
        
    """
    def get_dists_gn(self, gn0, order=2):
        dists = []
        for g in range(self.shape[1]):
            dists_g = []
            for i in range(1, self.shape[0]):
                for j in range(i):
                    gn_i = self[i, g, ]
                    gn_j = self[j]
                    diff_scl_gn = (gn_i - gn_j)/gn0
                    dists_g.append(scipy.linalg.norm(diff_scl_gn, order))
            dists.append(dists_g)
    """

