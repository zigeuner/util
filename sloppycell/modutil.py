"""
"""

from __future__ import division
from collections import OrderedDict as OD
import copy
import logging

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *

from util import butil, plotutil
reload(butil)
reload(plotutil)
from util.sloppycell import netutil, trajutil, ensutil
from util.sloppycell.mca import mcautil
reload(netutil)
reload(trajutil)
reload(ensutil)
reload(mcautil)


def make_mod(nets, datmap, **kwargs):
    """
    Input:
        datmap: eg, {'net1':{'A1':[1, 10], 'A2':[np.inf]}}
        kwargs: CV, sigma_min, sigma, fix_sf, add_noise
    """
    datmap = copy.deepcopy(datmap)
    
    expts = []
    count = 0
    for net in nets:
        varid2times = datmap.get(net.id)
        # collect steady-state varids
        ssvarids = []
        for varid in varid2times:
            if np.inf in varid2times[varid]:
                varid2times[varid].remove(np.inf)
                ssvarids.append(varid)
                
        # has steady-state data
        if len(ssvarids) > 0:
            ssvarvals = mcautil.get_ssvals(net, ssvarids)
            traj = trajutil.make_traj(net, ssvarids, [np.inf], ssvarvals)
            count += 1
            expt = trajutil.traj2expt(traj, net.id, exptid='expt_%d'%count, 
                                      **kwargs)
            expts.append(expt)
        
        times_unique = set([tuple(times) for times in varid2times.values()])
        # has time-series data
        if butil.flatten(times_unique):
            for times in times_unique: 
                varids = [varid for varid in varid2times 
                          if tuple(varid2times[varid])==times]
                traj = trajutil.get_traj(net, times=times, subvarids=varids,
                                         fill_traj=False)
                count += 1
                expt = trajutil.traj2expt(traj, net.id, exptid='expt_%d'%count, 
                                          **kwargs)
                expts.append(expt)
    mod = Model(expts, nets)
    return mod


def get_ss_datmap(net, concn=True, flux=True):
    """
    Make datmap for function make_mod using steady-state concentrations and/or
    fluxes. datmap: eg, {'net1':{'A1':[1, 10], 'A2':[np.inf]}}
    """
    varids = []
    if concn:
        varids += [sp.id for sp in net.species if not sp.is_constant]
    if flux:
        if not hasattr(net, 'fluxVars'):
            net = netutil.add_fluxes(net)
        varids += net.fluxVars.keys()
    times = [copy.copy([np.inf]) for i in range(len(varids))]
    datmap = {net.id: dict(zip(varids, times))}
    return datmap
    

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


def get_jacobian(mod, pvals=None, logparam=False):
    """
    Handle both dynamic and steady-state data. 
    In the case of steady-state data, calculate the response matrix.
    """
    mod2 = mod.copy()
    if pvals is None:
        pvals = mod2.params

    ## get residual keys corresponding to steady-state data
    reskeys_ss = []
    for reskey in mod2.residuals.keys():
        if isinstance(reskey, tuple):
            exptid, netid, varid, time = reskey 
            if time==np.inf:
                reskeys_ss.append(reskey)
                # it seems both places need to be changed
                del mod2.exptColl.get(exptid).data[netid][varid][time]
                mod2.residuals.remove_by_key(reskey)
                if mod2.exptColl.get(exptid).data[netid][varid] == {}:
                    del mod2.exptColl.get(exptid).data[netid][varid]
    for exptid, expt in mod2.exptColl.items():
        if expt.data.values() == [{}]:
            del mod2.exptColl[exptid]                
    
    
    ## calculate the response matrices
    for reskey in reskeys_ss:
        net = mod2.calcColl.get(reskey[1])
        R_concn = mcautil.get_concn_response_mat(net, pvals, normed_param=logparam)
        R_flux = mcautil.get_flux_response_mat(net, pvals, normed_param=logparam)
    
    ## get the rows in the response matrices
    jac_ss = KeyedList()
    for reskey in reskeys_ss:
        net = mod2.calcColl.get(reskey[1])
        varid = reskey[2]
        sigma = mod.residuals.get(reskey).ySigma
        if logparam:
            suffix = '_normed_param'
        else:
            suffix = ''
        # combine concentration and flux response matrix
        R = mcautil.MCAMatrix.vstack(getattr(net, 'concn_response_mat'+suffix),
                                     getattr(net, 'flux_response_mat'+suffix))
        jac_var = np.array(R.get_row(varid)).flatten()/sigma
        jac_ss.set(reskey, jac_var.tolist())
    
    ## calculate jacobian for dynamic data (if any) or priors (if any) 
    # if [rk for rk in mod2.residuals.keys() if isinstance(rk, tuple)]:
    # jac_other: jacobian of dynamic data or priors
    if logparam:
        jac_other = mod2.jacobian_log_params_sens_dyn(np.log(pvals))
    else:
        jac_other = mod2.jacobian_sens_dyn(pvals)
    
    jac = jac_ss + jac_other
    return jac
    

def sampling(mod, nstep, pvals=None, cutoff_singval=1e-6, seed=None, 
             step_scale=1, interval_recalc_hess=None, interval_print_step=None,
             raise_exception=True):
    """
    Input:
        
    """
    if pvals is None:
        pvals = mod.params
    pids = mod.calcColl.values()[0].optimizableVars.keys()

    #jac = get_jacobian(mod, pvals, logparam=True)
    #jtj = np.dot(np.transpose(jac), jac)
    
    
    try:
        ens, gs, r = Ensembles.ensemble_log_params(mod, pvals, None, steps=nstep, 
                                                   seeds=seed,
                                                   sing_val_cutoff=cutoff_singval,
                                                   step_scale=step_scale,
                                                   interval_recalc_hess=interval_recalc_hess,
                                                   interval_print_step=interval_print_step)
        # model, seed, r: custom attributes to be attached to pens (serialization?)
        pens = ensutil.ParameterEnsemble.from_sc_output(ens, gs, paramids=pids, 
                                                        model=mod, seed=seed, r=r)
        return pens
    except:
        if raise_exception:
            raise
        else:
            print "failed at seed: %d"%seed
            return None
    

def sampling_parallel(mod, ncore=None, seeds=None, b=0, k=1, **kwargs_sampling):
    """
    Input:
        ncore/seeds: at least one of them must be given
        b/k: thinning and decorrelating parameters
        kwargs_sampling:
            nstep
            pvals
            cutoff_singval
            step_scale
            interval_recalc_hess
            interval_print_step
    """
    f = lambda seed: sampling(mod, seed=seed, **kwargs_sampling)
    if seeds is None:
        seeds = range(ncore)
    #out = butil.map_parallel(f, seeds)
    out = map(f, seeds)
    pens_meta = ensutil.ParameterEnsemble.merge([pens.thinning(b=b, k=k) for pens
                                                 in out if pens is not None])
    return pens_meta


def jac2errs(jac, params_jac='', params_errs='', paramvals=None, 
             norm=True, datvarvals=None, CV=0.2, method='svd', cutoff=1e-6):
    """
    This function computes the standard deviations of (natural-log/log10)
    parameters from the given jacobian with respect to (natural-log/log10)
    parameters.
    
    Note: Eq. 10 in ~/work/CancerMetabolism/InterimReport/InterimReport.pdf 
          has the derivation.
    
    Input:
        jac: jacobian, with respect to:
             1. natural-log-parameters (params_jac='log',
                                        output of jacobian_log_params_sens)
             2. log10-parameters (params_jac='log10')
             3. parameters (params_jac='')
        params_jac: parameter options for the input jac (see above)
        params_errs: parameter options for the output errors (same as above)
        paramvals: needed for converting jac between the three types
        method: singular value decomposition ('svd') or 
                eigendecomposition ('eig'); they should be equivalent
        cutoff: cutoff for the singular values or eigenvalues 
                (Very small singular/eigen-values cause very large errors;
                 applying the cutoff makes all singular/eigen-values smaller 
                 than the cutoff equal to the cutoff)  
    Output: 
        errs: standard deviations of (natural-log-/log10-) parameters
    """
    jac = np.array(jac)
    
    if not norm:
        sigmas = np.array(datvarvals) * CV
        jac = jac / np.transpose([sigmas] * jac.shape[1])
    
    if paramvals is not None:
        paramvals = np.array(paramvals)
        
    ## update jac according to its input type and the output type of errors        
    if params_jac == '':
        if params_errs == 'log':
            jac = jac * paramvals
        if params_errs == 'log10':
            jac = jac * paramvals * np.log(10)
    if params_jac == 'log':
        if params_errs == '':
            jac = jac / paramvals
        if params_errs == 'log10':
            jac = jac * np.log(10)
    if params_jac == 'log10':
        if params_errs == '':
            jac = jac / paramvals / np.log(10)
        if params_errs == 'log':
            jac = jac / np.log(10)
    
    jac = np.matrix(jac)
    
    ## calculate errors
    if method == 'svd':
        U, S, Vh = np.linalg.svd(jac)
        SS_inv = np.matrix(np.diag(np.maximum(S, cutoff)**-2))
        errs = np.sqrt(np.diag(Vh.getT() * SS_inv * Vh))
    if method == 'eig':
        jtj = jac.getT() * jac
        evals, evecs = np.linalg.eig(jtj)
        D = np.matrix(np.diag(np.maximum(evals, evals[0]*cutoff)))
        V = np.matrix(evecs)
        errs = np.sqrt(np.diag(V * D.getI() * V.getT()))
        
    return errs
    

def plot_fitting(mod=None, expt=None, net_dat=None, net_fit=None, 
                 paramvals=None, plotvarids=None, normmap=None,
                 **kwargs):
    """
    Plot the residual of only one network.
    
    Input:
        mod: if given, has only one expt and one net. 
        expt: single network
        net_dat: for getting traj_dat that provides smooth curves of expt
        net_fit: for getting traj_fit; only one of mod and net should be given
        
        normmap: a map from varid to its value; 
                 if given, normalize expt and traj data by the values so that
                 all dynamics happen between 0 and 1
    """
    ## get net and expt
    if mod:
        net_fit = mod.GetCalculationCollection().values()[0]
        expt = mod.GetExperimentCollection().values()[0]
    varid2tser = expt.data.values()[0]  # only one network; tser: time series
    
    ## get traj_dat and traj_fit 
    if paramvals is not None:
        net_fit.update_optimizable_vars(paramvals)
    times = butil.flatten([data.keys() for data in varid2tser.values()])
    T = max(times) * 1.1
    if net_dat:
        traj_dat = Dynamics.integrate(net_dat, [0, T])
    else:
        traj_dat = None
    traj_fit = Dynamics.integrate(net_fit, [0, T])
        
    ## get plotvarids (if not given, use the intersection btw expt and traj)
    if plotvarids is None:
        plotvarids = [varid for varid in traj_fit.key_column.keys() 
                      if varid in varid2tser.keys()]
        
    ## collect to-be-plotted values
    trajs_x = []
    trajs_y = []
    trajs_err = []
    legends = []
    fmts = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
    for idx, varid in enumerate(plotvarids):
        tser = varid2tser[varid]
        times_sorted = sorted(tser.keys())
        data_sorted = butil.get_values(tser, times_sorted)
        trajs_x.append(times_sorted)
        if normmap:
            normval = normmap.get(varid)
            trajs_y.append([datum[0]/normval for datum in data_sorted])
            trajs_err.append([datum[1]/normval for datum in data_sorted])
        else:
            trajs_y.append([datum[0] for datum in data_sorted])
            trajs_err.append([datum[1] for datum in data_sorted])
        legends.append(varid+' (data point)')
        fmts.append('.'+colors[(idx)%len(colors)])
    # collect data curves (start again so that trajs_err can correspond to 
    # the first part of trajs_x and trajs_y)
    if traj_dat:
        for idx, varid in enumerate(plotvarids):
            trajs_x.append(traj_dat.timepoints)
            if normmap:
                normval = normmap.get(varid)
                trajs_y.append(traj_dat.get_var_traj(varid)/normval)
            else:
                trajs_y.append(traj_dat.get_var_traj(varid))
            legends.append(varid+' (data curve)')
            fmts.append('-'+colors[(idx)%len(colors)])
    # collect fitted curves (not collected at the same time with data curves 
    # so that they appear separately in the plot)
    for idx, varid in enumerate(plotvarids):
        trajs_x.append(traj_fit.timepoints)
        if normmap:
            normval = normmap.get(varid)
            trajs_y.append(traj_fit.get_var_traj(varid)/normval)
        else:
            trajs_y.append(traj_fit.get_var_traj(varid))
        legends.append(varid+' (fitted curve)')
        fmts.append('--'+colors[(idx)%len(colors)])
        
    if 'xmin' not in kwargs:
        kwargs['xmin'] = -T * 0.05

        
    ## plot
    plotutil.plot(trajs_x=trajs_x, trajs_y=trajs_y, trajs_err=trajs_err,
                  legends=legends, fmts=fmts, **kwargs)
    

def plot_errors(id2errors, paramid2val, norm=False, figtitle='', filepath=''):
    """
    """
    paramids = paramid2val.keys()
    paramvals = paramid2val.values()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    width = 1/(len(id2errors)+1)
    paramidxs = np.arange(len(paramids))
    
    colors = 'bgrcmy'
    for idx, errors in enumerate(id2errors.values()):
        if hasattr(errors, 'values'):
            errors = butil.get_values(errors, paramids)
        log10means = np.log10(paramvals)
        color = colors[idx % len(colors)]
        ax.bar(paramidxs+idx*width, log10means, width, color=color, 
               yerr=errors)
        
    ax.set_ylabel('log10(parameter estimates)')
    ax.set_xticks(paramidxs+width*len(id2errors)/2)
    ax.set_xticklabels(paramids, rotation='vertical')
    ax.legend(id2errors.keys())
    plt.subplots_adjust(bottom=0.2)
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    
def plot_fit_juxtapose(mod, paramvals_fit, folderpath=''):
    """
    """
    netid_x, netid_y = 'net_x', 'net_y'
    exptid_x, exptid_y = 'expt_x', 'expt_y'
    
    calcdat = mod.CalculateForAllDataPoints(paramvals_fit)
    #varids_x = mod.exptColl.get(exptid_x).data.values()[0].keys()
    spids = ['GLU', 'H6P', 'FBP', 'T3P', 'PG', 'PEP', 'PYR', '_6PG', 'R5P', 'SER', 'GLY']
    varids_x = [spid.lower()+'_l_x' for spid in spids] 
    varids_y = [spid.lower()+'_l_y' for spid in spids] 
    varids = zip(varids_x, varids_y)
    
    for idx, (varid_x, varid_y) in enumerate(varids):
        
        calcdat_x = calcdat.get(netid_x).get(varid_x) 
        exptdat_x = mod.get_expts().get(exptid_x).data.get(netid_x).get(varid_x)
        calcdat_y = calcdat.get(netid_y).get(varid_y)
        exptdat_y = mod.get_expts().get(exptid_y).data.get(netid_y).get(varid_y)
        
        if varid_x == '_3ps_l_x':
            continue
            exptdat_x.pop(15)
            exptdat_y.pop(15)
        #if varid_x == '_gly_l_x':
        #    exptdat_x.pop(15)
        #if varid_x == 'ser_l_x':
        #    exptdat_x.pop(15)
        
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        
        ts_calc_x = sorted(calcdat_x.keys())
        ax.plot(ts_calc_x, butil.get_values(calcdat_x, ts_calc_x), 'b')
        ts_expt_x = sorted(exptdat_x.keys())
        vals_x = np.array(butil.get_values(exptdat_x, ts_expt_x))[:,0]
        errs_x = np.array(butil.get_values(exptdat_x, ts_expt_x))[:,1]
        ax.errorbar(ts_expt_x, vals_x, errs_x, fmt=None, ecolor='b')
        
        ts_calc_y = sorted(calcdat_y.keys())
        ax.plot(ts_calc_y, butil.get_values(calcdat_y, ts_calc_y), 'g')
        ts_expt_y = sorted(exptdat_y.keys())
        vals_y = np.array(butil.get_values(exptdat_y, ts_expt_y))[:,0]
        errs_y = np.array(butil.get_values(exptdat_y, ts_expt_y))[:,1]
        ax.errorbar(ts_expt_y, vals_y, errs_y, fmt=None, ecolor='g')
            
        #duration = times_calc[-1] - times_calc[0]
        #ax.set_xlim(times_calc[0]-duration/10, times_calc[-1]+duration/10)
        if varid_x in ['gly_l_x', 'ser_l_x']:
            ax.legend(['5 mM', '0.5 mM'], loc='upper left', fontsize=12)
        if varid_x in ['_6pg_l_x', 'r5p_l_x', 'pep_l_x', 'pg_l_x', 't3p_l_x', 'pyr_l_x']:
            ax.legend(['5 mM', '0.5 mM'], loc='lower right', fontsize=12)
        if varid_x in ['h6p_l_x', 'fbp_l_x', 'glu_l_x']:
            ax.legend(['5 mM', '0.5 mM'], loc='center right', fontsize=12)
        ax.set_xticks([0, 2.5, 5, 10, 15])
        ax.set_xticklabels(['0','2.5','5','10','15'])
        ax.set_ylim(bottom=0)
        ax.yaxis.major.formatter.set_powerlimits((0,0))
        ax.ticklabel_format(style='sci', axis='y')
        
        spid = varid_x.replace('_l_x', '').upper()
        plt.title(spid)
        folderpath = folderpath.rstrip('/')
        plt.savefig('%s/fit%02d_%s.pdf'%(folderpath, idx+1, spid))
        plt.close()