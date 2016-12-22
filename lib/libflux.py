"""
Toy model:
A -> B -> C
       -> D
mbid2info: info=(concn, signal, signal2)
rxnid2info: info=(stoich, flux, flux2)

mbid2info = OD([('A', (1, 1,  2)),
                ('B', (1, 10, 20)),
                ('C', (1, 2,  1.5)),
                ('D', (1, 30, 20))])
rxnid2info = OD([('R1', ({'A':1},         2, 4)), 
                 ('R2', ({'A':-1, 'B':1}, 2, 4)),
                 ('R3', ({'B':-1, 'C':1}, 1, 3)),
                 ('R4', ({'C':-1},        1, 3)),
                 ('R5', ({'B':-1, 'D':1}, 1, 1)),
                 ('R6', ({'D':-1},        1, 1))])
"""

from __future__ import division
import copy
import cPickle

from collections import OrderedDict as OD
import sympy
import numpy as np
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *

import libtype
reload(libtype)
import libmca
reload(libmca)
import librxn
reload(librxn)
import libtraj
reload(libtraj)
import libmod
reload(libmod)

"""
def get_kinetic_net(mbid2info, rxnid2info, netid='net'):
    net = Network(netid)
    net.add_compartment('cell')
    for mbid, info in mbid2info.items():
        concn, signal, signal2 = info
        net.add_species(mbid, 'cell', concn, is_constant=True)
        net.species.get(mbid).signal = signal
        net.species.get(mbid).signal2 = signal2
    for rxnid, info in rxnid2info.items():
        stoich, flux, flux2 = info
        net.addReaction(rxnid, stoich, kineticLaw=str(flux))
        net.reactions.get(rxnid).kineticLaw2 = str(flux2)
    return net
"""
#net = get_kinetic_net(mbid2info, rxnid2info)


def x22y(name):
    """
    Convert a name of X/x to Y/y, or
            a name of Y/y to X/x.
            
    Input:
        name: a string
    """
    if 'x' in name or 'X' in name:
        return name.replace('X', 'Y').replace('x', 'y')
    if 'y' in name or 'Y' in name:
        return name.replace('Y', 'X').replace('y', 'x')


def xory(name, control):
    """
    """
    if ('x' in name or 'X' in name) and not control:
        return x22y(name)
    elif ('y' in name or 'Y' in name) and control:
        return x22y(name)
    else:
        return name


def get_net_afe(mbid2concn, rxnid2info, netid='net'):
    """
    AFE: absolute flux estimation. 
    
    Input: 
        mbid2concn, a dict mapping from species id to concentration;
            e.g., OD([('X1',1), 
                      ('X2',2)])
        rxnid2info, a dict mapping from reaction id to a tuple of
            stoichiometry matrix and flux;
            e.g., OD([('R1',({'X1':1}, 1)),
                      ('R2',({'X1':-1, 'X2':1}, 1)),
                      ('R3',({'X2':-1}, 1))])
    """
    net = Network(netid)
    net.add_compartment('cell')
    
    ## add labeled metabolite concentration (X_st) as species and
    ## pool size (X) as parameters
    for id_X, val_X in mbid2concn.items():
        net.add_species(id_X+'_st', 'cell', 0)
        net.add_parameter(id_X, val_X, is_optimizable=True)
        
    ## get the steadystate flux vectors
    rxnid2stoich = libtype.change_values(rxnid2info, lambda info: info[0])
    stoichmat = libmca.get_stoich_mat(rxnid2stoich=rxnid2stoich)
    ssfluxmat = libmca.get_ss_flux_mat(stoichmat=stoichmat)
    
    ## add independent fluxes
    # first get the estimated values of independent fluxes of control
    b = np.array([float(info[1]) for info in rxnid2info.values()])
    vals_Jind = np.linalg.lstsq(ssfluxmat, b)[0]
    ids_Jind = []  # independent flux ids
    for i in range(ssfluxmat.shape[1]):
        id_Jind = 'J' + str(i+1)
        ids_Jind.append(id_Jind)
        net.add_parameter(id_Jind, vals_Jind[i], is_optimizable=True)
        
    ## add dependent fluxes
    for rxnid, info in rxnid2info.items():
        stoich, val_Jdep = info
        # muls: multiplicities
        idx = ssfluxmat.colvarids.index(rxnid)
        muls = ['(%s)'%str(int(mul)) for mul in
                list(np.array(ssfluxmat[idx,:]).flatten())]
        
        id_Jdep = 'J' + rxnid
        net.add_parameter(id_Jdep, val_Jdep, 
                          is_constant=False, is_optimizable=False)
        assignmentrule = '+'.join(['*'.join([muls[i], ids_Jind[i]]) 
                                   for i in range(len(muls))])
        # simplify the assignment rule symbolically
        # e.g., '1*J1 + 0*J2' becomes 'J1'
        assignmentrule = str(sympy.simplify(assignmentrule))
        net.add_assignment_rule(id_Jdep, assignmentrule)
        
    ## add the rate rules
    # term: J * frac = J * X_st/X
    for id_X in mbid2concn.keys():
        terms_in, fluxs_out = [], []
        for rxnid, info in rxnid2info.items():
            stoich= info[0]
            id_Jdep = 'J'+rxnid
            if id_X in librxn.get_substrates(stoich):
                stoichcoef = stoich.get(id_X)
                # minus because stoichcoefs for substrates are negative
                fluxs_out.append(str(-stoichcoef)+'*'+id_Jdep)  
            if id_X in librxn.get_products(stoich):
                ids_substrate = [spid for spid in librxn.get_substrates(stoich)]
                frac_in = '*'.join(['%s_st/%s'%(id, id) for id in ids_substrate])
                # only the first rxn has no substrates or frac_in
                if frac_in == '':
                    frac_in = '1'
                stoichcoef = stoich.get(id_X)
                terms_in.append(str(stoichcoef)+'*'+id_Jdep+'*'+frac_in)
        # if terms_in or fluxs_out remain empty, 
        # then the metabolite does not have any source or sink respectively
        if terms_in == []:
            terms_in = ['0']
        if fluxs_out == []:
            fluxs_out = ['0']
        frac_out = '%s_st/%s' % (id_X, id_X)
        # simplify the terms
        # e.g., '1*JR1*1' becomes 'JR1'
        terms_in = [str(sympy.simplify(term_in)) for term_in in terms_in]
        fluxs_out = [str(sympy.simplify(flux_out)) for flux_out in fluxs_out]
        raterule = '(%s)-(%s)*%s' % ('+'.join(terms_in), 
                                     '+'.join(fluxs_out), 
                                     frac_out)
        net.add_rate_rule(id_X+'_st', raterule)
    
    return net
        

def get_net_rfe(mbid2info, rxnid2info, netid='net_x', control=True):
    """
    RFE: relative flux-change estimation
    
    Naming conventions:
        X: species/concentration (of control)
        pX: ionization strength
        rX: change ratio between control and condition
        x: mass spec signal of X; x = X * pX
        x_st: st for star, mass spec signal of labeled X
        Y: species/concentration of condition; Y = X * rX
        y: mass spec signal of Y; y = Y * pX = x * rX
        y_st: st for star, mass spec signal of labeled Y
    """
    if netid is None and not control:
        netid = 'net_y'
    net = Network(netid)
    net.add_compartment('cell')
    
    ## add the species and parameters (optimizable/independent and dependent)
    for id_X, info in mbid2info.items():
        val_X, val_x, val_y = info  # ctrl concn, ctrl signal, condition signal
        id_x = id_X.lower()
        id_x_st = id_x + '_st'
        id_pX = 'p' + id_X
        id_rX = 'r' + id_X
        val_pX = val_x / val_X
        val_rX = val_y / val_x
        net.add_parameter(id_X, val_X, is_optimizable=True)
        net.add_parameter(id_pX, val_pX, is_optimizable=True)
        net.add_parameter(id_rX, val_rX, is_optimizable=True)
        
        net.add_species(xory(id_x_st, control), 'cell', 0)
        if control:
            net.add_parameter(id_x, val_X*val_pX, 
                              is_constant=False, is_optimizable=False)
            net.add_assignment_rule(id_x, '%s*%s'%(id_X, id_pX))
        else:
            net.add_parameter(x22y(id_x), val_X*val_rX*val_pX, 
                              is_constant=False, is_optimizable=False)
            net.add_assignment_rule(x22y(id_x), '%s*%s*%s'%(id_X, id_rX, id_pX))
            
    """    
        ratio = signal2 / signal  # ratio=condition/control
        p = signal / concn
        net.add_parameter(id_concn, concn, is_optimizable=True)
        net.add_parameter(id_p, p, is_optimizable=True)
        net.add_parameter(id_ratio, ratio, is_optimizable=True)
        if control:
            net.add_species(id_signal_st, 'cell', 0)
            net.add_parameter(id_signal, concn*p, is_constant=False, is_optimizable=False)
            net.add_assignment_rule(id_signal, '%s*%s'%(id_concn, id_p))
        if not control:
            net.add_species(id_signal_st+'2', 'cell', 0)
            #net.add_parameter(id_concn+'2', concn*ratio, is_constant=False, is_optimizable=False)
            #net.add_assignment_rule(id_concn+'2', '%s*%s'%(id_concn, id_ratio))
            net.add_parameter(id_signal+'2', concn*ratio*p, is_constant=False, is_optimizable=False)
            net.add_assignment_rule(id_signal+'2', '%s*%s*%s'%(id_concn, id_ratio, id_p))
    """
    
    ## get the steadystate flux vectors
    rxnid2stoich = libtype.change_values(rxnid2info, lambda info: info[0])
    stoichmat = libmca.get_stoich_mat(rxnid2stoich=rxnid2stoich)
    ssfluxmat = libmca.get_ss_flux_mat(stoichmat=stoichmat)
    
    ## add independent fluxes
    # first get the estimated values of independent fluxes of control
    b_x = np.array([float(info[1]) for info in rxnid2info.values()])
    vals_Jind_x = np.linalg.lstsq(ssfluxmat, b_x)[0]  
    ids_Jind_x = []  # independent flux ids
    for i in range(ssfluxmat.shape[1]):
        id_Jind_x = 'J' + str(i+1) + 'x'
        ids_Jind_x.append(id_Jind_x)
        net.add_parameter(id_Jind_x, vals_Jind_x[i], is_optimizable=True)
    
    ## add the change ratios of independent fluxes
    # first get the estimated values of independent fluxes of condition
    b_y = np.array([float(info[2]) for info in rxnid2info.values()])
    vals_Jind_y = np.linalg.lstsq(ssfluxmat, b_y)[0]
    ids_rJ = []
    for idx, id_Jind_x in enumerate(ids_Jind_x):
        id_rJ = 'r' + id_Jind_x.rstrip('x')
        ids_rJ.append(id_rJ)
        val_rJ = vals_Jind_y[idx] / vals_Jind_x[idx]
        net.add_parameter(id_rJ, val_rJ, is_optimizable=True)
        
    ## add dependent fluxes
    for rxnid, info in rxnid2info.items():
        stoich, val_Jdep_x, val_Jdep_y = info
        # muls: multiplicities
        idx = ssfluxmat.colvarids.index(rxnid)
        muls = ['(%s)'%str(int(mul)) for mul in
                list(np.array(ssfluxmat[idx,:]).flatten())]
        if control:
            id_Jdep = 'J' + rxnid + 'x'
            net.add_parameter(id_Jdep, val_Jdep_x, 
                              is_constant=False, is_optimizable=False)
            assignmentrule = '+'.join(['*'.join([muls[i], ids_Jind_x[i]]) 
                                       for i in range(len(muls))])
        else:
            id_Jdep = 'J' + rxnid + 'y'
            net.add_parameter(id_Jdep, val_Jdep_y, 
                              is_constant=False, is_optimizable=False)
            assignmentrule = '+'.join(['*'.join([muls[i], ids_Jind_x[i], ids_rJ[i]])
                                       for i in range(len(muls))])
        # simplify the assignment rule symbolically
        # e.g., '1*J1 + 0*J2' becomes 'J1'
        assignmentrule = str(sympy.simplify(assignmentrule))
        net.add_assignment_rule(id_Jdep, assignmentrule)

    ## add the rate rules
    for id_X in mbid2info.keys():
        terms_in, terms_out = [], []
        for rxnid, info in rxnid2info.items():
            stoich, val_Jdep_x, val_Jdep_y = info
            id_Jdep = xory('J'+rxnid+'x', control)
            if id_X in librxn.get_substrates(stoich):
                stoichcoef = stoich.get(id_X)
                terms_out.append(str(-stoichcoef)+'*'+id_Jdep)
            if id_X in librxn.get_products(stoich):
                ids_x = [s.lower() for s in librxn.get_substrates(stoich)]
                frac_in = '*'.join(['%s_st/%s'%(xory(id_x, control), 
                                                xory(id_x, control)) 
                                    for id_x in ids_x])
                # only the first rxn has no substrates or frac_in  
                if frac_in == '':
                    frac_in = '1'
                stoichcoef = stoich.get(id_X)
                terms_in.append(str(stoichcoef)+'*'+id_Jdep+'*'+frac_in)
        # if terms_in or terms_out remain empty, 
        # then the metabolite does not have any source or sink respectively
        if terms_in == []:
            terms_in = ['0']
        if terms_out == []:
            terms_out = ['0']
        id_signal_st = xory(id_X.lower()+'_st', control)
        frac_out = '%s/%s' % (id_signal_st, id_signal_st.replace('_st', ''))
        # simplify ...
        terms_in = [str(sympy.simplify(term_in)) for term_in in terms_in]
        terms_out = [str(sympy.simplify(term_out)) for term_out in terms_out]
        raterule = '%s*(%s-(%s)*%s)' % ('p'+id_X, 
                                        '+'.join(terms_in), 
                                        '+'.join(terms_out), 
                                        frac_out)
        net.add_rate_rule(id_signal_st, raterule)
    
    return net

    
def afe(net, paramid2val=None, times=None, ntimes=None, T=1000, cutoff_variation=0.95, 
        datvarids=None, net_r=None, paramid2val_r=None, sigma=1,
        CV=0.2, add_noise=False, tol=1e-9):
    """
    Absolute flux estimation (AFE) has three main cases:
    1. Partial data, reduced network:
        common situation, common practice (we argue against)
    2. Partial data, full network:
        common situation, suggested practice
    3. All data, full network:
        ideal situation, natural practice
        
    Partial or all data depends on the input datvarids;
    reduced or full network depends on the input net_r (if it is given). 
    
    Input: 
        paramid2val:
        net_r: reduced network, which if given, is used for estimation
        datvarids: variable ids for which we pretend to have data
        times: seq or int. If seq, the sampling times; if int, the number 
               of sampling times
        T: the maximal integration time;
           a parameter used for variation-based sampling times determination
        cutoff_variation: the cutoff for the total variation up to time T;
                          a parameter used for variation-based sampling
                          times determination
        paramid2val_r: a dict to update the parameter values of net_r
                       which ultimately *only* changes the paramvals_trial
                       of the estimation
        tol: sometimes the dynamics is stiff (e.g., a very small pool size)
             and needs a more stringent tolerance
             
    Output:
        paramvals_fit:
        stdevs:
    """
    if datvarids is None:
        datvarids = net.species.keys()
    
    ## copy and update nets
    net = net.copy()
    net_cp = net.copy()
    if paramid2val is not None:
        net.set_var_vals(paramid2val)
    if net_r:
        net_r = net_r.copy()
        if paramid2val_r:
            net_r.set_var_vals(paramid2val_r)
        
    ## generate data and make mod
    ndynvar = len(net.dynamicVars)
    if ntimes:  # determine sampling times by variation 
        traj = Dynamics.integrate(net, [0, T], rtol=[tol]*ndynvar, 
                                  atol=[tol]*ndynvar, fill_traj=True)
        times = libtraj.get_sampling_times(traj, datvarids, n=ntimes,
                                           cutoff=cutoff_variation)
        times = np.concatenate(([0], times))
    traj = Dynamics.integrate(net_cp, times, rtol=[tol]*ndynvar, 
                              atol=[tol]*ndynvar, fill_traj=False)
    if net_r:
        netid = net_r.id
        net_mod = net_r
    else:
        netid = net.id
        net_mod = net
    expt = libtraj.traj2expt(traj, datvarids=datvarids, netid=netid, 
                             CV=CV, add_noise=add_noise)
    mod = Model([expt], [net_mod])

    ## estimate parameters
    paramvals_true = np.array([p.value for p in net_mod.optimizableVars])
    # sigma is one order of magnitude (lognormal)
    paramvals_trial = paramvals_true *\
        np.random.lognormal(sigma=sigma, size=len(paramvals_true))
    paramvals_fit = Optimization.fmin_lm_log_params(mod, paramvals_trial, 
                                                    maxiter=20, disp=False)
    
    ## estimate standard deviations
    jac = mod.jacobian_log_params_sens(np.log(paramvals_fit))
    stdevs = libmod.get_parameter_stdevs(jac, log10=False, singval_cutoff=1e-9)
    
    return paramvals_fit, stdevs


def rfe(net_x, net_y, paramid2val=None, times=None, ntimes=None, T=1000, cutoff_variation=0.95, 
        datvarids_x=None, net_r_x=None, net_r_y=None, paramid2val_r=None,
        sigma=1, CV=0.2, add_noise=False, tol=1e-9):
    """
    Relative flux-change estimation (RFE)
    
    net_x and net_y should have the same optimizable parameters;
    net_r_x and net_r_y should have the same optimizable parameters, but
    not necessarily the same as net_x/net_y. 
    
    Input:
        times: one of times and ntimes must be given
        ntimes: int; if given, determine sampling times from variation
        datvarids_x: data variable ids for net_x, assuming the corresponding
                     data variables for net_y is also available;
                     if None, defaults to all species in net_x
        paramid2val_r: a dict to update the parameter values of net_r_x
                       and net_r_y, which ultimately *only* changes
                       the paramvals_trial of the estimation
        sigma: a parameter to control the noise added to paramvals_trial
    """
    ## copy and update nets
    net_x = net_x.copy()
    net_y = net_y.copy()
    net_x_cp = net_x.copy()
    net_y_cp = net_y.copy()
    if paramid2val:
        net_x.set_var_vals(paramid2val)
        net_y.set_var_vals(paramid2val)
        net_x_cp.set_var_vals(paramid2val)
        net_y_cp.set_var_vals(paramid2val)
    if net_r_x and net_r_y:
        net_r_x = net_r_x.copy()
        net_r_y = net_r_y.copy()
        if paramid2val_r:
            net_r_x.set_var_vals(paramid2val_r)
            net_r_y.set_var_vals(paramid2val_r)
        
    ## generate data and make mod
    nsp = len(net_x.species)
    datvarids_y = [x22y(id) for id in datvarids_x]
    if bool(times) == bool(ntimes):
        raise StandardError('Only times or ntimes should be specified.')
    if ntimes:
        traj_x = Dynamics.integrate(net_x, [0, T], rtol=[tol]*nsp, 
                                    atol=[tol]*nsp, fill_traj=True)
        traj_y = Dynamics.integrate(net_y, [0, T], rtol=[tol]*nsp, 
                                    atol=[tol]*nsp, fill_traj=True)
        curvtraj_x = libtraj.traj2curvtraj(traj_x)
        curvtraj_y = libtraj.traj2curvtraj(traj_y)
        times_x = libtraj.get_sampling_times(curvtraj_x, datvarids_x, n=ntimes,
                                             cutoff=cutoff_variation)
        times_y = libtraj.get_sampling_times(curvtraj_y, datvarids_y, n=ntimes,
                                             cutoff=cutoff_variation)
        times_x = np.concatenate(([0], times_x))
        times_y = np.concatenate(([0], times_y))
        #print 'times_x', times_x
        #print 'times_y', times_y
    if times:
        times_x = times_y = times
    traj_x = Dynamics.integrate(net_x_cp, times_x, rtol=[tol]*nsp, 
                                atol=[tol]*nsp, fill_traj=False)
    traj_y = Dynamics.integrate(net_y_cp, times_y, rtol=[tol]*nsp, 
                                atol=[tol]*nsp, fill_traj=False)
    if net_r_x and net_r_y:
        netid_x = net_r_x.id
        netid_y = net_r_y.id
        net_mod_x = net_r_x
        net_mod_y = net_r_y
        paramvals_true = np.array([p.value for p in net_r_x.optimizableVars])
    else:
        netid_x = net_x.id
        netid_y = net_y.id
        net_mod_x = net_x
        net_mod_y = net_y
        paramvals_true = np.array([p.value for p in net_x.optimizableVars])
    if datvarids_x is None:
        datvarids_x = net_x.species.keys()
    expt_x = libtraj.traj2expt(traj_x, datvarids=datvarids_x, netid=netid_x, 
                               CV=CV, add_noise=add_noise)
    expt_y = libtraj.traj2expt(traj_y, datvarids=datvarids_y, netid=netid_y, 
                               CV=CV, add_noise=add_noise)
    mod = Model([expt_x, expt_y], [net_mod_x, net_mod_y])

    ## estimate parameters
    paramvals_trial = paramvals_true *\
        np.random.lognormal(sigma=sigma, size=len(paramvals_true))
    paramvals_fit = Optimization.fmin_lm_log_params(mod, paramvals_trial, 
                                                    maxiter=15, disp=False)

    ## estimate standard deviations
    jac = mod.jacobian_log_params_sens(np.log(paramvals_fit))
    stdevs = libmod.get_parameter_stdevs(jac, log10=False, singval_cutoff=1e-9)
    
    return paramvals_fit, stdevs


def recompute_bad_spots(func, means, stdevs, id_xaxis, id_yaxis, 
                        vals_xaxis, vals_yaxis, badspots, tol=1e-2, 
                        filepath_means=None, filepath_stdevs=None):
    """
    A function that recomputes the miscomputed estimates (due to 
    badly chosen trial parameter values). 
    
    Input: 
        func: a function that takes in paramid2val and 
              spits out (mean, stdev)
        means and stdevs: np.ndarray of shape (len(vals_xaxis), len(vals_yaxis))
                          and whose bad spots are to be replaced
        id_xaxis: e.g., 'X1', 'X'
        id_yaxis: e.g., 'rX1', 'q'
        badspots: a list of (val_xaxis, val_yaxis) tuples
        tol: a parameter for getting the indices of val_xaxis & val_yaxis
             in vals_xaxis & vals_yaxis
    
    Output:
        Recomputed means and stdevs
    """
    vals_xaxis, vals_yaxis = np.array(vals_xaxis), np.array(vals_yaxis)
    for badspot in badspots:
        val_xaxis, val_yaxis = badspot
        idxs_xaxis = np.where(np.abs(vals_xaxis-val_xaxis)/vals_xaxis<tol)[0]
        
        idxs_yaxis = np.where(np.abs(vals_yaxis-val_yaxis)/vals_yaxis<tol)[0]
        if len(idxs_xaxis) > 1:
            raise StandardError("can't locate the idx of xaxis")
        if  len(idxs_yaxis) > 1:
            raise StandardError("can't locate the idx of yaxis")
        print idxs_xaxis, idxs_yaxis
        idx_xaxis = idxs_xaxis[0]
        idx_yaxis = idxs_yaxis[0]
        paramid2val = {id_xaxis: val_xaxis, id_yaxis: val_yaxis}
        mean, stdev = func(paramid2val)
        print mean, stdev
        print idx_xaxis, idx_yaxis
        means[idx_xaxis, idx_yaxis] = mean
        stdevs[idx_xaxis, idx_yaxis] = stdev
    if filepath_means:
        np.save(filepath_means, means)
    if filepath_stdevs:
        np.save(filepath_stdevs, stdevs)
    return means, stdevs
    

def rfeinfo2afeinfo(mbid2info_rfe, rxnid2info_rfe, control=True):
    """
    A function that ...
    
    mbid2info = OD([
        ('X1', (1, 1, 3)),
        ('X2', (1, 1, 3))])

    rxnid2info = OD([
        ('R1', ({'X1':1}, 1, 2)),
        ('R2', ({'X1':-1, 'X2':1}, 1, 2)),
        ('R3', ({'X2':-1}, 1, 2))])
    """
    mbid2concn_afe, rxnid2info_afe = OD(), OD()
    for mbid, info in mbid2info_rfe.items():
        if control:
            concn = info[1]
        else:
            concn = info[2]
        mbid2concn_afe[mbid] = concn
    for rxnid, info in rxnid2info_rfe.items():
        stoich = info[0]
        if control:
            J = info[1]
        else:
            J = info[2]
        rxnid2info_afe[rxnid] = (stoich, J)
    return mbid2concn_afe, rxnid2info_afe
    

def print_net(net):
    """
    """
    print "Species:\n\t", [(sp.id, sp.value) for sp in net.species]
    print "Optimizable Parameters:\n\t", [(v.id, v.value) for v in net.optimizableVars]
    print "(Non-optimizable) Parameters:\n\t", [(p.id, p.value) for p in net.parameters 
                                                if not p.is_optimizable]
    print "Assignment Rules:\n\t", net.assignmentRules.items()
    print "Rate Rules:\n\t", net.rateRules.items()
    
    
class FluxEstimates(np.ndarray):
    """
    Estimates of both AFE and RFE. 
    1-dim, 2-dim or 4-dim. 
    """
    
    def __new__(cls, dat=None, id2vals=None, varid='', 
                nets_dat=None, nets_est=None):
        """
        Terminology:
            id: name of the index, e.g., 'X', 'dim1'
            idx: always integers
            mulidx: multi-index, a tuple of integers
            val: the value of an idxid corresponding to an index
            vals: all the values of an idxid
            
        Input: 
            id2vals: an od mapping from id to vals; 
                     e.g., OD([('X', [0.1, 1, 10]), 
                               ('q', [0.3, 0.5, 0.7])]) 
            nets_dat: networks used for generating data
            nets_est: networks used for estimations
        """
        if id2vals is None:
            raise StandardError("id2vals has to be provided")
        idxshape = tuple([len(vals) for vals in id2vals.values()])
        if dat is not None:
            if dat.shape != idxshape:
                raise StandardError("shape not agreed")
        else:
            dat = np.zeros(idxshape)
        obj = np.asarray(dat).view(cls)
        obj.id2vals = id2vals
        obj.varid = varid
        obj.nets_dat = nets_dat
        obj.nets_est = nets_est
        return obj


    def __array__finalize__(self, obj):
        if obj is None: 
            return
        self.id2vals = getattr(obj, 'id2vals', None)
        self.varid = getattr(obj, 'varid', None)
        self.nets_dat = getattr(obj, 'nets_dat', None)
        self.nets_est = getattr(obj, 'nets_est', None)
        
    
    def get(self, id2val):
        """
        Input:
            id2val: can be a mapping or a sequence;
                    if a sequence, the order has to follow self.id2vals
        """
        if not hasattr(id2val, 'items'):  # convert to a mapping  
            id2val = OD(zip(self.id2vals.keys(), id2val))
        mulidx = tuple([libtype.index(self.id2vals[id], val) 
                        for id, val in id2val.items()])
        return self[mulidx]
    
    
    def set(self, id2val, value):
        """
        Input:
            id2val: can be a mapping or a sequence;
                    if a sequence, the order has to follow self.id2vals
        """
        if not hasattr(id2val, 'items'):  # convert to a mapping  
            id2val = OD(zip(self.id2vals.keys(), id2val))
        mulidx = tuple([libtype.index(self.id2vals[id], val)
                        for id, val in id2val.items()])
        self[mulidx] = value
    
        
    def normalize(self, trueval, log10=False):
        """
        """
        dat = self / trueval
        varid = self.varid + '_norm'
        if log10:
            dat = np.log10(dat)
            varid = 'log10_' + varid 
        return FluxEstimates(dat, id2vals=self.id2vals, varid=varid, 
                             nets_dat=self.nets_dat, nets_est=self.nets_est)
        
    
    def save(self, filepath):
        """
        """
        dat = (np.array(self), self.id2vals, self.varid, 
               self.nets_dat, self.nets_est)
        fh = open(filepath, 'w')
        cPickle.dump(dat, fh)
        fh.close()
    
    
    @staticmethod
    def load(filepath):
        fh = open(filepath)
        arr, id2vals, varid, nets_dat, nets_est = cPickle.load(fh)
        fh.close()
        return FluxEstimates(arr, id2vals=id2vals, varid=varid, 
                             nets_dat=nets_dat, nets_est=nets_est)
    
    
    def plot(self, latex_text=True, cbar_exp10=False, figtitle='', filepath=''):
        """
        A wrapper. 
        1-dim, 2-dim, and 4-dim
        """
        if self.ndim == 1:
            self.plot_1d()
        if self.ndim == 2:
            self.plot_2d(latex_text=latex_text, cbar_exp10=cbar_exp10,
                         figtitle=figtitle, filepath=filepath)
        if self.ndim == 4:
            self.plot_4d(latex_text=latex_text, cbar_exp10=cbar_exp10,
                         figtitle=figtitle, filepath=filepath)
    
    
    def plot_1d(self):
        pass
    
    
    def plot_2d(self, latex_text=True, cbar_exp10=False, figtitle='', filepath=''):
        """
        Input:
            self: a data *matrix* whose heatmap is to be plotted; 
                 self.shape == (len(vals_x), len(vals_y)).
                 Note that the heatmap (1) has vals_x corresponding to rows, and
                                               vals_y corresponding to columns,
                                       (2) has vals_y increasing upward.
                 These two requirements correspond to the transpose and flipud 
                 operations of mat, respectively. 
                 (Also the flipping [::-1] of vals_y)
            id2vals: an od of two items giving the x/y labels and values in order
        """
        (xlabel, ylabel), (vals_x, vals_y) =\
            self.id2vals.keys(), self.id2vals.values()
        if latex_text:
            xlabel = '$' + xlabel + '$'
            ylabel = '$' + ylabel + '$'
            #figtitle = '$' + figtitle + '$'
            
        plt.rc('text', usetex=True)    
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        heat = ax.imshow(np.flipud(self.transpose()), interpolation='hamming')
        ax.set_xticks(np.arange(self.shape[0]))
        ax.set_yticks(np.arange(self.shape[1]))
        # remove the extra "0" in the scientific notation
        #format = lambda s: ('%.1e'%s).replace('e+0', 'e+').replace('e-0', 'e-')
        ax.set_xticklabels([libtype.format(val_x) for val_x in vals_x], rotation='vertical', size=10)
        ax.set_yticklabels(['%.2f'%val_y for val_y in vals_y[::-1]], size=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, size=14)
        ax.set_title(figtitle, position=(0.5, 1), size=13)
        
        fig.subplots_adjust(bottom=0.15, right=0.85)
        ## add color bar
        cax = fig.add_axes([0.85, 0.12, 0.03, 0.8])
        cbar = fig.colorbar(heat, cax=cax)
        # remove xticklabels
        cbar.ax.set_xticks([])
        if cbar_exp10:
            yticklabels = cbar.ax.get_yticklabels()
            yticklabels = [float(l.get_text()[1:-1]) for l in yticklabels]
            yticklabels = np.round(np.power(10, yticklabels), 2)
            cbar.ax.set_yticklabels(yticklabels)
        # adjust the size of yticklabels
        cbar.ax.tick_params(axis='y', labelsize=10)
        
        plt.savefig(filepath, dpi=300)
        plt.close()
        plt.rc('text', usetex=False)
    
    
    def plot_4d(self, latex_text=True, cbar_exp10=False, figtitle='', filepath=''):
        """
        """
        from libtype import format
            
        id_v1, id_v2, id_v3, id_v4 = self.id2vals.keys() 
        vals_v1, vals_v2, vals_v3, vals_v4 = self.id2vals.values()    
        if latex_text:
            id_v1, id_v2, id_v3, id_v4 = '$'+id_v1+'$', '$'+id_v2+'$',\
                                         '$'+id_v3+'$', '$'+id_v4+'$',    
        min, max = np.min(self), np.max(self)
        
        plt.rc('text', usetex=True)
        fig = plt.figure(dpi=300, figsize=(4*len(vals_v1), 3*len(vals_v2)))
        #plt.title(figtitle)
    
        for idx_v2, val_v2 in enumerate(vals_v2[::-1]): 
            for idx_v1, val_v1 in enumerate(vals_v1):
                i = idx_v2*len(vals_v1) + idx_v1 + 1
                ax = fig.add_subplot(len(vals_v2), len(vals_v1), i)
                #print idx_v1, len(vals_v2), idx_v2
                mat = self[idx_v1, len(vals_v2)-idx_v2-1]
                heat = ax.imshow(np.flipud(mat.transpose()), 
                                 interpolation='hamming', vmin=min, vmax=max)
                ax.set_xticks([])
                ax.set_yticks([])
    
                xticks = np.arange(0, len(vals_v3), 2)
                xticklabels = [format(val_v3) for val_v3 in vals_v3[::2]]
                yticks = np.arange(0, len(vals_v4), 2)
                #yticklabels = [format(val_v2) for val_v2 in vals_v2[::2]]
                yticklabels = [format(val_v4) for val_v4 in vals_v4[::-2]]
                if idx_v2 == 0:
                    ax.set_xticks(xticks)
                    ax.xaxis.set_ticks_position('top')
                    ax.set_xticklabels(xticklabels, rotation='vertical', size=12)
                    ax.set_xlabel(format(val_v1), fontsize=14)
                    ax.xaxis.set_label_position('top')
                if idx_v2 == len(vals_v2)-1:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, rotation='vertical', size=12)
                    ax.set_xlabel(format(val_v1), fontsize=14)
                if idx_v1 == 0:
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels, size=12)
                    ax.set_ylabel(format(val_v2), rotation='horizontal', fontsize=14)
                if idx_v1 == len(vals_v1)-1:
                    ax.set_yticks(yticks)
                    ax.yaxis.set_ticks_position('right')
                    ax.set_yticklabels(yticklabels, size=12)
                    ax.yaxis.set_label_position('right')
                    ax.set_ylabel(format(val_v2), rotation='horizontal', fontsize=14)
    
        fig.subplots_adjust(right=0.85, hspace=0, wspace=0.05)
        ## add color bar
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(heat, cax=cax)
        # highlight zero
        xlim = cax.get_xlim()
        y = (0-min) / (max-min)
        cax.plot(xlim, [y, y], 'r-', linewidth=5)
        # remove xticklabels
        cbar.ax.set_xticks([])
        # adjust the size of yticklabels
        cbar.ax.tick_params(axis='y', labelsize=20)
        
        plt.savefig(filepath, dpi=300)
        plt.close()
        plt.rc('text', usetex=False)
    
    
    def find_badspots(self):
        pass
    
    
def prod(id2vals, vals, d=2):
    """
    """
    pass