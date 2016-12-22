"""
"""
from __future__ import division

from collections import OrderedDict as OD
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *

import libmca
reload(libmca)
import libtype


def jacobian_log_params_sens(mod, logparamvals):
    """
    A replacemnet of SloppyCell.Model_mod.Model.jacobian_log_params_sens
    """
    paramvals = np.exp(logparamvals)
    jac = jacobian_sens(mod, paramvals)
    logjac = jac.copy()
    logjac.update(np.asarray(jac) * paramvals)
    return logjac


def jacobian_sens(mod, paramvals):
    """
    A replacement of SloppyCell.Model_mod.Model.jacobian_sens
    Return a keyedlist
    """
    # require that all nets have the same optimizable parameters
    optvarids_nets = [net.optimizableVars.keys() 
                      for net in mod.calcColl.values()]
    if not libtype.all_same(optvarids_nets):
        raise StandardError("nets do not have the same set of\
                                 optimizable variables.")
    
    # require that all times are np.inf
    times = np.array([res.xVal for res in mod.residuals.values() 
                      if hasattr(res, 'xVal')])
    if np.any(times != np.inf):
        raise StandardError("not all times in the residuals are np.inf.") 
    
    # initialize jac_scaled, the jacobian scaled by sigmas
    jac_scaled = KeyedList(zip(mod.residuals.keys(), 
                               [None]*len(mod.residuals)))
    
    for (reskey, res) in mod.residuals.items():
        if jac_scaled.get(reskey) is None:  # the value has not been filled yet
            net = mod.calcColl.get(res.calcKey)
            varid = res.yKey
        
            # if the variable is a steady-state concentration
            if varid in net.dynamicVars.keys():
                jac_concn = libmca.get_concn_response_mat(net, paramvals)
                # fill all possible entries of jac_scaled
                for rowvarid in jac_concn.rowvarids:
                    for (reskey, res) in mod.residuals.items():
                        if (hasattr(res, 'calcKey') and res.calcKey==net.id) and\
                            (hasattr(res, 'yKey') and res.yKey==rowvarid):
                            rowidx = jac_concn.rowvarids.index(rowvarid)
                            # scaled by sigma
                            sens_scaled = np.array(jac_concn[rowidx]/res.ySigma)
                            sens_scaled = sens_scaled.flatten().tolist()
                            jac_scaled.set(reskey, sens_scaled)
                            
            # if the variable is a steady-state flux
            if hasattr(net, 'fluxVars') and varid in net.fluxVars.keys():
                jac_flux = libmca.get_flux_response_mat(net, paramvals)
                # fill all possible entries of jac_scaled
                for rowvarid in jac_flux.rowvarids:
                    for (reskey, res) in mod.residuals.items():
                        if (hasattr(res, 'calcKey') and res.calcKey==net.id) and\
                            (hasattr(res, 'yKey') and res.yKey==rowvarid):
                            rowidx = jac_flux.rowvarids.index(rowvarid)
                            # scaled by sigma
                            sens_scaled = np.array(jac_flux[rowidx]/res.ySigma)
                            sens_scaled = sens_scaled.flatten().tolist()
                            jac_scaled.set(reskey, sens_scaled)
                            
    return jac_scaled

    """
    nets = mod.GetCalculationCollection()  # a list of nets 
    residuals = mod.GetResiduals()
    jac_scaled = KeyedList()
    for idx, net in enumerate(nets):
        expt = mod.exptColl[idx]
        jac_net = libmca.get_concentration_response_matrix(net, paramvals)
        for (resid, res) in residuals.items():
            # two kinds of residuals: one from data and one from priors;
            # only data residuals have ids in tuple
            if isinstance(resid, tuple) and res.calcKey == net.id:
                rowidx = net.dynamicVars.keys().index(res.yKey)
                # scaled by sigma
                sens_scaled = jac_net[rowidx] / res.ySigma
                sens_scaled = np.array(sens_scaled).flatten().tolist()
                jac_scaled.set(resid, sens_scaled)
    return jac_scaled
    """    

def add_priors(mod, paramid2val=None, factor=100):
    """
    factor: a parameter indicating the variation of the parameter;
            with 0.95 probability that the parameter is going to be between
            mean*factor and mean/factor
    """
    if paramid2val is None:
        paramid2val = mod.params
    for paramid, paramval in paramid2val.items():
        res = Residuals.PriorInLog('prior_'+paramid, paramid, np.log(paramval),
                                   np.log(np.sqrt(factor)))
        mod.AddResidual(res)
    return mod


def get_parameter_stdevs(J, singval_cutoff=1e-6, log10=True, paramids=None):
    """
    Eq. 10 in the interim report
    
    Input:
        J: jacobian with respect to log-parameter (natural log);
           an output of libmod.jacobian_log_params_sens
        log10: whether the output is with respect to log10-parameters
    Output: 
        stdevs: standard deviations of log(10)-parameters
    """
    U, S, Vh = np.linalg.svd(J)
    SS_inv = np.matrix(np.diag(np.maximum(S, singval_cutoff)**-2))
    stdevs = np.sqrt(np.diag(Vh.transpose() * SS_inv * Vh))
    if log10:  # scale to log10-param variation (decreased)
        stdevs = stdevs * np.log10(np.e)
    if paramids is not None:
        stdevs = OD(zip(paramids, stdevs))
    return stdevs


def get_parameter_stdevs2(J, singval_cutoff=1e-6, log10=True, paramids=None):
    """
    Input:
        J: jacobian with respect to log-parameter (natural log);
           an output of libmod.jacobian_log_params_sens
        log10: whether the output is with respect to log10-parameters
    Output: 
        stdevs: standard deviations of log(10)-parameters
    """
    JtJ = np.dot(np.transpose(J), J)
    evals, evecs = np.linalg.eig(JtJ)
    D = np.matrix(np.diag(np.maximum(evals, evals[0] * singval_cutoff)))
    V = np.matrix(evecs)
    stdevs = np.sqrt(np.diag(V * D.getI() * V.getT()))
    if log10:  # scale to log10-param variation (decreased)
        stdevs = stdevs * np.log10(np.e)
    if paramids is not None:
        stdevs = OD(zip(paramids, stdevs))
    return stdevs


def plot_stdevs(id2stdevs, paramid2val, norm=False, figtitle='', filepath=''):
    """
    """
    paramids = paramid2val.keys()
    paramvals = paramid2val.values()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    width = 1/(len(id2stdevs)+1)
    paramidxs = np.arange(len(paramids))
    
    colors = 'bgrcmy'
    for idx, stdevs in enumerate(id2stdevs.values()):
        if hasattr(stdevs, 'values'):
            stdevs = libtype.get_values(stdevs, paramids)
        log10means = np.log10(paramvals)
        color = colors[idx % len(colors)]
        ax.bar(paramidxs+idx*width, log10means, width, color=color, 
               yerr=stdevs)
        
    ax.set_ylabel('log10(parameter estimates)')
    ax.set_xticks(paramidxs+width*len(id2stdevs)/2)
    ax.set_xticklabels(paramids, rotation='vertical')
    ax.legend(id2stdevs.keys())
    plt.subplots_adjust(bottom=0.2)
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    

def get_timeseries(mod, netid, varid):
    """
    """
    exptid2expt = mod.get_expts()
    for expt in exptid2expt.values():
        if netid in expt.data.keys():
            return expt.data.get(netid).get(varid)
    
    
def plot_fit(mod, paramvals_fit, dirpath=''):
    """
    """
    calcdat = mod.CalculateForAllDataPoints(paramvals_fit)
    for netid, calcdat_net in calcdat.items():
        for varid, calcdat_net_var in calcdat_net.items():
            exptdat_net_var = get_timeseries(mod, netid, varid)
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            times_calc = sorted(calcdat_net_var.keys())
            ax.plot(times_calc, 
                    libtype.get_values(calcdat_net_var, times_calc))
            times_expt = sorted(exptdat_net_var.keys())
            ys = np.array(libtype.get_values(exptdat_net_var, times_expt))[:,0]
            yerrs = np.array(libtype.get_values(exptdat_net_var, times_expt))[:,1]
            ax.errorbar(times_expt, ys, yerrs, fmt=None)
            duration = times_calc[-1] - times_calc[0]
            ax.set_xlim(times_calc[0]-duration/10, times_calc[-1]+duration/10)
            title = '%s_%s'%(netid, varid)
            plt.title(title)
            plt.savefig('%s/%s.png'%(dirpath, title), dpi=300)
            
            
def plot_fit_juxtapose(mod, paramvals_fit, folderpath=''):
    """
    """
    netid, netid2 = 'net', 'net2'
    exptid, exptid2 = 'expt_net', 'expt_net2'
    
    calcdat = mod.CalculateForAllDataPoints(paramvals_fit)
    #for varid in ['dhap_st']:#net.species.keys():
    for varid in ['glu_st',
 'h6p_st',
 '_6pg_st',
 'fbp_st',
 'dhap_st',
 'gap_st',
 '_3pg_st',
 'pep_st',
 'pyr_st',
 'lac_st']:
        varid2 = varid + '2'
        
        calcdat_net = calcdat.get(netid).get(varid) 
        exptdat_net = mod.get_expts().get(exptid).data.get(netid).get(varid)
        calcdat_net2 = calcdat.get(netid2).get(varid2) 
        exptdat_net2 = mod.get_expts().get(exptid2).data.get(netid2).get(varid2)
        
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        
        times_calc = sorted(calcdat_net.keys())
        ax.plot(times_calc, libtype.get_values(calcdat_net, times_calc), 'b')
        times_expt = sorted(exptdat_net.keys())
        ys = np.array(libtype.get_values(exptdat_net, times_expt))[:,0]
        yerrs = np.array(libtype.get_values(exptdat_net, times_expt))[:,1]
        ax.errorbar(times_expt, ys, yerrs, fmt=None, ecolor='b')
        
        times_calc2 = sorted(calcdat_net2.keys())
        ax.plot(times_calc2, libtype.get_values(calcdat_net2, times_calc2), 'g')
        times_expt2 = sorted(exptdat_net2.keys())
        ys2 = np.array(libtype.get_values(exptdat_net2, times_expt2))[:,0]
        yerrs2 = np.array(libtype.get_values(exptdat_net2, times_expt2))[:,1]
        ax.errorbar(times_expt2, ys2, yerrs2, fmt=None, ecolor='g')
            
        duration = times_calc[-1] - times_calc[0]
        ax.set_xlim(times_calc[0]-duration/10, times_calc[-1]+duration/10)
        ax.legend(['5000uM', '500uM'])
        
        mbid = varid.rstrip('_st').upper()
        plt.title(mbid)
        plt.savefig('%s/%s.png'%(folderpath, mbid), dpi=300)