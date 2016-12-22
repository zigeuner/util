"""
Some utility functions for the SloppyCell Trajectory instances.
"""

import copy
from collections import OrderedDict as OD
import scipy as sp 
import numpy as np
import matplotlib.pyplot as plt
from SloppyCell.ReactionNetworks import *
import libtype
reload(libtype)
import libplot
reload(libplot)
import libgeo
reload(libgeo)


def get_subset(traj, t0=None, idx0=None, keys=None):
    traj = copy.deepcopy(traj)
    if idx0:
        traj.timepoints = traj.timepoints[idx0:]
        traj.values = traj.values[idx0:, :]
    if keys:
        traj = traj.copy_subset(keys)
    return traj


def plot_trajs(traj, figsize=(8,6), plotvarids=None, legend=True, 
               xlabel='', ylabel='', ts_mark=None,
               styles=None, figtitle='', filepath=''):
    """
    Plot the trajectories of given variables.
    
    Input:
        traj: a SloppyCell Trajectory instance
        varids: a list of variable ids, e.g., ['RuBP', 'v_r41139']
        log10: if True, no *zeros* or *negative numbers* allowed in traj
        ts_mark: if given, mark the timepoints, used for highlighting
                 sampling times (from equipartition of variation)     
        filepath: a string of path for the figure to be saved
    """
    if plotvarids is None:
        plotvarids = traj.key_column.keys()
    idxs = [traj.key_column.get(plotvarid) for plotvarid in plotvarids]
    ts = traj.timepoints
    dat = traj.values[:,idxs]
    
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
        ax.legend(plotvarids, prop={'size': 9})
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    
"""    
def plot_time_series_with_errors(traj, CV, varids=None, plot_traj=True, filepath=''):
    if varids is None:
        varids = traj.key_column.keys()
    idxs = [traj.key_column.get(varid) for varid in varids]
    times = traj.timepoints
    timeseries_all = traj.values[:,idxs]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for timeseries in timeseries_all.transpose():
        ax.errorbar(range(1,1+len(times)), timeseries, yerr=timeseries*CV, fmt='.')
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
    from libflux import x22y
    
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


def traj2curvtraj(traj, varids=None):
    """
    Given a SloppyCell traj object, construct and return another 
    SloppyCell traj object, that contains the curvature trajectories
    of the original trajectories. 
    """
    curvtraj = traj.copy_subset(varids)
    values = []
    for varid in curvtraj.key_column.keys():
        values.append(libgeo.est_curvatures(traj.timepoints, 
                                            curvtraj.get_var_traj(varid)))
    curvtraj.values = np.array(values).transpose()
    return curvtraj


def get_sampling_times(curvtraj, varids=None, cutoff=0.95, n=5):
    """
    Return the sampling timepoints according to the equipartition
    of the areas under the curvature curves. 
    
    Input:
        cutoff: float between 0 and 1;
                how much area (variation) is retained
        n: number of timepoints
    """
    curvtraj = curvtraj.copy_subset(varids)
    ts = curvtraj.timepoints
    varid2areas = OD()
    for varid in curvtraj.key_column.keys():
        curvs = curvtraj.get_var_traj(varid)
        areas = np.concatenate(([0], sp.integrate.cumtrapz(curvs, ts)))
        varid2areas[varid] = areas
    areas_total = np.sum(varid2areas.values(), axis=0)
    maxarea = areas_total[-1] * cutoff
    areas_sampled = np.arange(maxarea/n, maxarea+1e-6, maxarea/n)
    idxs_sampled = [libtype.get_closest_index(areas_total, area_sampled)
                    for area_sampled in areas_sampled]
    ts_sampled = [ts[idx_sampled] for idx_sampled in idxs_sampled]
    return ts_sampled


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
    """
    varvals = [var.value for var in net.dynamicVars] +\
              [var.value for var in net.assignedVars]
    traj = Trajectory_mod.Trajectory(net)
    traj.timepoints = np.array([time])
    traj.values = np.array(varvals).reshape((1,len(varvals)))
    return traj


def dynvarvals2traj(dynvarvals, net, time=np.inf):
    """
    """
    net_new = net.copy()
    net_new.updateVariablesFromDynamicVars(dynvarvals, time)
    return net2traj(net_new, time=time)


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


def traj2expt(traj, netid, datvarids=None, exptid=None, CV=0.05, 
              fix_sf=True, add_noise=False):
    """
    expt = Experiment('expt1')
    data = {'net1':{'data1': {0: (0, 0.037),
                              2: (0.16575, 0.025),
                              },
                    'data2': {0: (1, 0.084),
                              2: (0.9275, 0.046),
                              }
                   }
           }
    expt.set_data(data)


    len(times) has to match len(traj)
    noise: 
        True: noisy data
        False: perfect 
    
    """
    if datvarids is None:
        datvarids = traj.key_column.keys()
    varids = traj.key_column.keys()
    data_net = {}
    for i in range(len(traj.key_column)):
        varid = varids[i]
        if varid in datvarids:  # get values of vars for which we have data
            time2vals = {}
            for j in range(len(traj.timepoints)):
                time = traj.timepoints[j]
                val = traj.values[j, i]
                se = val * CV  # se: standard error
                # if the value is zero, standard error is set to CV
                if val == 0:
                    se = CV
                if add_noise:
                    # np.abs to enforce positivity
                    val = np.abs(val + np.random.randn() * se)
                time2vals[time] = (val, se)
            data_net[varid] = time2vals
    if exptid is None:
        exptid  = 'expt_' + netid
    expt = Experiment(exptid)
    expt.set_data({netid: data_net})
    if fix_sf:
        expt.set_fixed_sf(dict.fromkeys(varids, 1))
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

