"""
Some utility functions for the SloppyCell Trajectory instances.
"""

import copy
from collections import OrderedDict as OD

import scipy as sp 
import numpy as np
import matplotlib.pyplot as plt
from SloppyCell.ReactionNetworks import *

from util2 import butil, plotutil
from util2.sloppycell.mca import mcautil
reload(butil)
reload(plotutil)
reload(mcautil)


def plot_trajs(trajs, **kwargs): 
               
               
               #legend=True, 
               #xlabel='', ylabel='', ts_mark=None, fmts=None, 
               
               #figtitle='', filepath=''):
    """
    Plot the trajectories of given variables.
    
    Input:
        traj: a SloppyCell Trajectory instance
        plotvarids: a list of variable ids, e.g., ['RuBP', 'v_r41139']
        kwargs: other arguments of plotutil.plot, and its docstring is
                attached below.
            
    """
    if hasattr(trajs, '__iter__'):
        trajs_x = []
        trajs_y = [] 
        for traj in trajs:
            trajs_x.extend([traj.timepoints.tolist()]*len(traj.key_column))
            trajs_y.extend(traj.values.T.tolist())
    else:
        traj = trajs
        trajs_x = traj.timepoints
        trajs_y = traj.values.T

    plotutil.plot(trajs_x, trajs_y, **kwargs)
    
    """
    
    if plotvarids is None:
        plotvarids = traj.key_column.keys()
    idxs = [traj.key_column.get(plotvarid) for plotvarid in plotvarids]
    ts = traj.timepoints
    dat = traj.values[:,idxs].T
    
    
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    if styles:
        for ys, style in zip(dat.transpose(), styles):
            ax.plot(ts, ys, style)
    else:
        ax.plot(ts, dat)
    h = ax.get_ylim()[1] / 20
    if ts_mark is not None:
        for t in ts_mark:
            ax.plot([t, t], [0, h], '-k', markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(plotvarids, prop={'size': 12}, loc='lower right')
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
    """
plot_trajs.__doc__ += plotutil.plot.__doc__
    
    
"""    
def plot_time_series_with_errors(traj, cv, varids=None, plot_traj=True, filepath=''):
    if varids is None:
        varids = traj.key_column.keys()
    idxs = [traj.key_column.get(varid) for varid in varids]
    times = traj.timepoints
    timeseries_all = traj.values[:,idxs]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for timeseries in timeseries_all.transpose():
        ax.errorbar(range(1,1+len(times)), timeseries, yerr=timeseries*cv, fmt='.')
        if plot_traj:
            pass  # need improvement
    ax.set_xticklabels(times)
    ax.set_xlim(right=len(times)*1.1)
    plt.legend(varids, prop={'size':9})
    plt.savefig(filepath, dpi=300)
    plt.close()
""" 

    
def plot_trajs_xny(traj_x, traj_y, figsize=(8,6), subplots=True, 
                   vertical=True, plotvarids_x=None, xlabel='', ylabel='', 
                   ts_mark_x=None, ts_mark_y=None, 
                   plotvarnames=None, figtitle='', filepath=''):
    """
    Plot the trajectories of both control (x) and condition (y). 
    
    Input:
        plotvarids_x: assume the corresponding varids in traj_y are also 
                      available; conversion is done using libflux.x22y
        ts_mark: if given, mark the timepoints, used for highlighting
                 sampling times (from equipartition of variation)
    """
    from fluxest import x22y
    
    if plotvarids_x is None:  # plot all the species
        plotvarids_x = [varid for varid in traj_x.key_column.keys()
                        if varid.endswith('_st')]
    if plotvarnames:
        plotvarid2name = dict(zip(plotvarids_x, plotvarnames))
    # generate one fig with different metabolites vertically or
    # horizontally juxtaposed
    if subplots:
        if vertical:
            figsize = (8, 6*len(plotvarids_x))
            figdim = [len(plotvarids_x), 1]
        else:
            figsize = (8*len(plotvarids_x), 6)
            figdim = [1, len(plotvarids_x)]
        plt.rc('text', usetex=False)  
        fig = plt.figure(dpi=300, figsize=figsize)
        for idx, plotvarid_x in enumerate(plotvarids_x):
            ax = fig.add_subplot(*figdim+[idx+1])
            plotvarid_y = x22y(plotvarid_x)
            traj_var_x = traj_x.get_var_traj(plotvarid_x)
            traj_var_y = traj_y.get_var_traj(plotvarid_y)
            ax.plot(traj_x.timepoints, traj_var_x)
            ax.plot(traj_y.timepoints, traj_var_y)
            if plotvarnames:
                plotvarname = plotvaridx2name[plotvarid_x]
                legend = [plotvarname+' control', plotvarname+' condition']
            else:
                legend = [plotvarid_x, plotvarid_y]
            ax.legend(legend)
        plt.savefig(filepath, dpi=300)
        plt.close()
    # generate different figs for different metabolites
    else:
        fig = plt.figure(dpi=300, figsize=figsize)
        ax = fig.add_subplot(111)
        for plotvarid_x in plotvarids_x:
            plotvarid_y = x22y(plotvarid_x)
            traj_var_x = traj_x.get_var_traj(plotvarid_x)
            traj_var_y = traj_y.get_var_traj(plotvarid_y)

            ob = ax.plot(traj_x.timepoints, traj_var_x, '-', 
                         label=plotvarid_x)
            ax.plot(traj_y.timepoints, traj_var_y, '--', 
                    color=ob[0].properties()['color'])
        ax.plot([0], [0], 'k-', label='control')
        ax.plot([0], [0], 'k--', label='condition')
        
        h = ax.get_ylim()[1] / 20
        if ts_mark_x is not None:
            for t in ts_mark_x:
                ax.plot([t, t], [0, h], '-k', lw=1) 
        if ts_mark_y is not None:
            for t in ts_mark_y:
                ax.plot([t, t], [0, h], '--k', dashes=(2,1), lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.subplots_adjust(left=0.2)
        """
            if plotvarnames:
                plotvarname = plotvaridx2name[plotvarid_x]
                legend = [plotvarname+' control', plotvarname+' condition']
                filename = plotvarname + '.png'
            else:
                legend = [plotvarid_x, plotvarid_y]
                filename = str(plotvarid_x) + '.png'
        """
        plt.title(figtitle)
        plt.savefig(filepath, dpi=300)
        plt.close()


def get_subtraj(traj, t0=None, idx0=None, varids_subset=None):
    """
    """
    subtraj = traj.copy_subset()
    if idx0:
        subtraj.timepoints = traj.timepoints[idx0:]
        subtraj.values = traj.values[idx0:, :]
    if keys:
        subtraj = subtraj.copy_subset(varids_subset)
    return subtraj


def is_variable_constant(traj, varid):
    """
    """
    traj_var = traj.get_var_traj(varid)
    return np.array_equal(traj_var, np.ones(len(traj_var))*traj_var[0])


def is_traj_steady_state(traj, tol=1e-9):
    """
    """
    derivs = (traj.values[-1,:] - traj.values[-2,:]) / \
             (traj.timepoints[-1] - traj.timepoints[-2])
    if np.max(np.abs(derivs)) < tol:
        return True
    else:
        return False


def net2traj(net, time=np.inf):
    """
    Deprecated. Replaced by make_traj.
    """
    varvals = [var.value for var in net.dynamicVars] +\
              [var.value for var in net.assignedVars]
    traj = Trajectory_mod.Trajectory(net)
    traj.timepoints = np.array([time])
    traj.values = np.array(varvals).reshape((1,len(varvals)))
    return traj


def dynvarvals2traj(dynvarvals, net, time=np.inf):
    """
    Deprecated. Replaced by make_traj.
    """
    net_new = net.copy()
    net_new.updateVariablesFromDynamicVars(dynvarvals, time)
    return net2traj(net_new, time=time)


def make_traj(net, varids, times, vals):
    """
    Make a traj from scratch.
    """
    traj = Trajectory_mod.Trajectory(net).copy_subset(varids)
    traj.timepoints = np.array(times)
    traj.values = np.array(vals).reshape((len(times),len(varids)))
    return traj
    

def merge_trajs(traj1, traj2, net):
    """
    A replacement of the Trajectory.merge method, which does not sort time, and
    Trajectory.append method, which requires traj1.timepoints be before 
    traj2.timepoints
    """
    ts1 = traj1.timepoints
    dat1 = np.append(np.reshape(ts1, (len(ts1),1)), traj1.values, 1)
    ts2 = traj2.timepoints
    dat2 = np.append(np.reshape(ts2, (len(ts2),1)), traj2.values, 1)
    dat = np.append(dat1, dat2, 0)
    dat_sorted = sorted(zip(dat[:,0], dat[:,1:]), key=lambda x: x[0])
    ts_sorted = [row[0] for row in dat_sorted]
    values_sorted = np.array([row[1] for row in dat_sorted])
    # create a new traj instance to incorporate only the timepoints and values
    # of traj1 and traj2
    traj = Trajectory_mod.Trajectory(net)
    traj.timepoints = ts_sorted
    traj.values = values_sorted
    return traj


def _get_idxs_time(times, subtimes, cutoff=1e-6):
    idxs = []
    for time in subtimes:
        idx = list(butil.get_indices(times, time, cutoff=cutoff))
        if idx:
            idxs.append(idx[0])
        else:
            return
    return idxs
        

def get_vals_var(traj, varid, times=None):
    """
    if times is given and is not the same as traj.timepoints,
    calculate values at the given times through interpolation 
    """
    vals_var = traj.get_var_traj(varid)
    
    if times is not None:
        idxs = _get_idxs_time(traj.timepoints, times)
        if idxs:
            vals_var = traj.values[idxs, traj.key_column.get(varid)]
        else:  # interpolation
            if isinstance(varid, tuple):
                fid = 'f_%s'%'_'.join(varid)
            else:
                fid = 'f_%s'%varid
            
            try:
                f = getattr(traj, fid)
            except AttributeError:
                f = sp.interpolate.InterpolatedUnivariateSpline(traj.timepoints, 
                                                                vals_var)
                setattr(traj, fid, f)
            vals_var = f(times)

    return vals_var


def get_traj_times(traj, times):
    """
    """
    traj_times = traj.copy_subset()
    traj_times.timepoints = np.array(times)
    vals_times = np.array([get_vals_var(traj, varid, times=times) 
                           for varid in traj.key_column.keys()]).T
    traj_times.values = vals_times
    return traj_times
    

def get_traj(net, times, sen=False, tol=None,
             subvarids=None, interpolate=False, 
             store=False, **kwargs_integrate):
    """
    A convenient wrapper of Dynamics.integrate(_sensitivity) that: 
    1) avoids repeated integrations;
    2) makes sure the returned traj are for the input times;
    3) handles both traj and sensitivity traj;
    4) gets a subset of vars.
    
    Input:
        sen: if True, return the sensitivity traj
        interpolate: if net has traj, then force interpolation
        store: if True, attach traj to net as an attribute
        
    """
    times = sorted(times)
    
    if sen:
        trajid = 'sentraj'
        func = Dynamics.integrate_sensitivity
    else:
        trajid = 'traj'
        func = Dynamics.integrate
        
    try:
        traj = getattr(net, trajid)
        if not butil.true_if_close(times, traj.timepoints): 
            if interpolate:
                traj = get_traj_times(traj, times)
            else:
                raise ValueError("Different Times.")
    except (AttributeError, ValueError):
        if tol:
            tol = [tol] * len(net.dynamicVars)
        if times[0] == 0:
            traj = func(net, times, rtol=tol, **kwargs_integrate)
        else:
            traj = func(net, [0]+times, rtol=tol, **kwargs_integrate)
            traj = get_traj_times(traj, times)

            """
            alltimes = traj.timepoints
            idxs = np.where((alltimes>=times[0]) & (alltimes<=times[-1]))[0]
            times = alltimes[idxs]
            traj = get_traj_times(traj, times)
            """
        if store:
            setattr(net, trajid, traj)
        
    if subvarids:
        traj = traj.copy_subset(subvarids)
        
    return traj


"""
def get_sens_var_param(sentraj, varid, paramid, times=None):
    sens = sentraj.get_var_traj((varid, paramid))
    if times is not None and\
            not butil.true_if_close(times, sentraj.timepoints, 1e-6):
        try:
            f = getattr(traj, 'f_%s_%s'%(varid, paramid))
        except AttributeError:
            f = sp.interpolate.InterpolatedUnivariateSpline(sentraj.timepoints, 
                                                            sens)
            setattr(sentraj, 'f_%s_%s'%(varid, paramid), f)
        sens = f(times)
    return sens
"""


def traj2expt(traj, netid, exptid=None, 
              datvarids=None, times=None, jitter_time=None, 
              cv=0.2, sigma_min=0.1, sigma=None, 
              fix_sf=True, add_noise=False):
    """
    This function constructs a SloppyCell Experiment object from a SloppyCell
    Trajectory object. 
    
    A SloppyCell Experiment object contains data of the following structure:
    expt = Experiment('expt1')
    data = {'net1':{'var1': {0: (0, 0.037),
                             2: (0.16575, 0.025)},
                    'var2': {0: (1, 0.084),
                             2: (0.9275, 0.046)}
                   }
           }
    expt.set_data(data)

    Input: 
        netid: used as 'net1' in the example
        exptid: used as 'expt1' in the example; if not given, use 'expt_'+netid 
        times: if given and is different from traj.timepoints, 
               *get data at the given times using interpolation*
        jitter_time: if given, keep all the times in traj.timepoints by adding
                     a little noise
        cv: coefficient of variation 
        fix_sf: whether fix scale factor
        add_noise: whether add noise to the values (to make the estimation 
                   more realistic)
    Output:
        expt: a SloppyCell Experiment object
    """
    ## get datvarids and times
    if datvarids is None:
        datvarids = traj.key_column.keys()
    if times is None:
        times = traj.timepoints
    if jitter_time:
        times = times * np.random.lognormal(0, jitter_time, len(times))
    
    ## get data
    data = {}
    for varid in datvarids:
        # get values
        vals_var = get_vals_var(traj, varid, times)
        # sometimes the values get close to 0 but negative
        vals_var = np.abs(vals_var)
        # sigmas proportional to values (by cv)  
        if cv:  
            sigmas_var = vals_var * cv
            # set to sigma_min if falling below it 
            # prob. most realistic; the DEFAULT
            if sigma_min:  
                sigmas_var = np.maximum(sigmas_var, sigma_min)
        # constant sigmas
        if sigma:  
            sigmas_var = sigma * np.ones(len(vals_var))
        if add_noise:
            vals_var = np.abs(vals_var +\
                              np.random.randn(len(vals_var))*sigmas_var)
        # make data
        data_var = zip(vals_var, sigmas_var)
        time2data_var = dict(zip(times, data_var))
        if varid == 'dG_R2_cp':
            dG = traj.get_var_traj(varid)[0]
            time2data_var = {1: (dG, -1*dG*cv)}
        data[varid] = time2data_var
    #import ipdb
    #ipdb.set_trace()
    ## get expt
    if exptid is None:
        exptid  = 'expt_' + netid
    expt = Experiment(exptid)
    expt.set_data({netid: data})
    if fix_sf:
        expt.set_fixed_sf(dict.fromkeys(datvarids, 1))
    return expt


"""
def add_steady_states(traj, dynvarssvals):
    ""
    Return a new traj instance with steady-state information added. 
    time: scipy.inf
    ""
    
    traj2 = traj.copy_subset()
    traj2.timepoints = np.append(traj.timepoints, np.inf)
    dynvarssvals = np.reshape(dynvarssvals, (1, len(dynvarssvals)))
    traj2.values = np.append(traj.values, dynvarssvals, 0)
    return traj2
"""


def Calculate(net, vars, paramvals=None):
    """
    A replacement of SloppyCell.ReactionNetworks.Network_mod.Network.Calculate
    to include steady-state computations.
    """
    times = set([])
    ## flag 'has_inf' detects if there is scipy.inf in times;
    ## if yes, remove scipy.inf from the list, marks the flag True
    ## and calls a routine to calculate steady-state concentrations
    has_inf = False
    for var, ts in vars.items():
        if net.variables.has_key(var):
            if sp.inf in ts:
                has_inf = True
                ts.remove(sp.inf)
            times = set.union(times, set(ts))
        else:
            raise ValueError('Unknown variable %s requested from network %s'
                             % (var, net.id))

    times = sorted(list(times))
    if times:
        net.update_optimizable_vars(paramvals)
        traj1 = get_traj(net, times, fill_traj=False)
    else:
        traj1 = Trajectory_mod.Trajectory(net)  # empty traj
    if has_inf:
        ssvals = mcautil.get_ssvals(net, paramvals=paramvals)
        traj2 = make_traj(net, net.dynamicVars.keys()+net.assignedVars.keys(), 
                          [np.inf], ssvals)
    else:
        traj2 = Trajectory_mod.Trajectory(net)  # empty traj
    traj = merge_trajs(traj1, traj2, net)
    net.trajectory = traj

