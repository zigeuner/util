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

from collections import OrderedDict as OD
import sympy
import numpy as np

from SloppyCell.ReactionNetworks import *

import libmca
import librxn


def get_kinetic_net(mbid2info, rxnid2info, netid='net'):
    """
    """
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

#net = get_kinetic_net(mbid2info, rxnid2info)


def get_flux_net(mbid2info, rxnid2info, netid='net', control=True):
    """
    """
    if netid is None and not control:
        netid = 'net2'
    net = Network(netid)
    net.add_compartment('cell')
    
    ## add the species and parameters (optimizable/independent and dependent)
    for mbid, info in mbid2info.items():
        concn, signal, signal2 = info
        id_concn = mbid.upper()
        id_signal = mbid.lower()
        id_signal_st = id_signal + '_st'
        id_p = 'p' + id_concn.lstrip('_')
        id_ratio = 'r' + id_concn.lstrip('_')
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
        
    ## get the steadystate flux vectors
    net0 = get_kinetic_net(mbid2info, rxnid2info, netid='')
    stoichmat = libmca.get_stoich_mat(net0, dynamic=False)
    ssfluxmat = libmca.get_ss_flux_mat(net0)
    
    ## add independent fluxes
    # first get the estimated values of independent fluxes of control
    b = np.array([float(info[1]) for info in rxnid2info.values()])
    indepfluxvals = np.linalg.lstsq(ssfluxmat, b)[0]
    indepfluxids = []
    for i in range(ssfluxmat.shape[1]):
        indepfluxid = 'J' + str(i+1)
        indepfluxids.append(indepfluxid)
        net.add_parameter(indepfluxid, indepfluxvals[i], is_optimizable=True)
        
    ## add the change ratios of independent fluxes
    # first get the estimated values of independent fluxes of condition
    b2 = np.array([float(info[2]) for info in rxnid2info.values()])
    indepfluxvals2 = np.linalg.lstsq(ssfluxmat, b2)[0]
    fluxratioids = []
    for idx, indepfluxid in enumerate(indepfluxids):
        fluxratioid = 'r' + indepfluxid
        fluxratioids.append(fluxratioid)
        ratio = indepfluxvals2[idx] / indepfluxvals[idx]
        net.add_parameter(fluxratioid, ratio, is_optimizable=True)
        
    ## add dependent fluxes
    for rxnid, info in rxnid2info.items():
        stoich, flux, flux2 = info 
        if control:
            fluxid = 'J' + rxnid.lstrip('_')
        else:
            fluxid = 'J' + rxnid.lstrip('_') + '2'
        net.add_parameter(fluxid, flux, is_constant=False, is_optimizable=False)
        # muls: multiplicities
        idx = ssfluxmat.colvarids.index(rxnid)
        muls = ['(%s)'%str(int(mul)) for mul in
                list(np.array(ssfluxmat[idx,:]).flatten())]
        if control:
            assignmentrule = '+'.join(['*'.join([muls[i], indepfluxids[i]]) 
                                       for i in range(len(muls))])
        else:
            assignmentrule = '+'.join(['*'.join([muls[i], indepfluxids[i], fluxratioids[i]])
                                       for i in range(len(muls))])
        # simplify the assignment rule symbolically
        # e.g., '1*J1 + 0*J2' becomes 'J1'
        assignmentrule = str(sympy.simplify(assignmentrule))
        net.add_assignment_rule(fluxid, assignmentrule)

    ## add the rate rules
    for mbid in mbid2info.keys():
        
        terms_in, terms_out = [], []
        for rxnid, info in rxnid2info.items():
            stoich, flux, flux2 = info
            if control:
                fluxid = 'J' + rxnid.lstrip('_')
            else:
                fluxid = 'J' + rxnid.lstrip('_') + '2'
            if mbid in librxn.get_substrates(stoich):
                terms_out.append(fluxid)
            if mbid in librxn.get_products(stoich):
                ids_signal = [s.lower() for s in librxn.get_substrates(stoich)]
                if control:
                    frac_in = '*'.join(['%s_st/%s'%(s, s) for s in ids_signal])
                else:
                    frac_in = '*'.join(['%s_st2/%s2'%(s, s) for s in ids_signal])
                # only the first rxn has no substrates or frac_in  
                if frac_in == '':
                    frac_in = '1'
                terms_in.append(fluxid+'*'+frac_in)
                
        # if terms_in or terms_out remain empty, 
        # then the metabolite does not have any source or sink respectively
        if terms_in == []:
            terms_in = ['0']
        if terms_out == []:
            terms_out = ['0']
            
        if control:
            id_signal_st = mbid.lower() + '_st'
        else:
            id_signal_st = mbid.lower() + '_st2'
        frac_out = '%s/%s' % (id_signal_st, id_signal_st.replace('_st', ''))
        raterule = '%s*(%s-(%s)*%s)' % ('p'+mbid.upper().lstrip('_'), 
                                        '+'.join(terms_in), 
                                        '+'.join(terms_out), 
                                        frac_out)
        net.add_rate_rule(id_signal_st, raterule)
    
    return net


