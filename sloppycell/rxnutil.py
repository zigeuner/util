"""
"""

from collections import OrderedDict as OD
import re

import sympy
import numpy as np

from SloppyCell.ReactionNetworks.Reactions import Reaction

from util2 import butil
reload(butil)


def get_substrates(stoich, multi=False):
    """
    Input: 
        multi:
    """
    for spid, stoichcoef in stoich.items():
        # could be ...
        stoichcoef = int(float(stoichcoef))  
    
    subs = [[spid]*abs(int(float(stoichcoef))) for (spid, stoichcoef) 
            in stoich.items() if stoichcoef<0]
    subs = butil.flatten(subs)
    if not multi:
        subs = list(set(subs))
    subs = sorted(subs)  # make spids appear in order
    return subs


def get_products(stoich, multi=False):
    pros = [[spid]*abs(stoichcoef) for (spid, stoichcoef)
            in stoich.items() if stoichcoef>0]
    pros = butil.flatten(pros)
    if not multi:
        pros = list(set(pros))
    pros = sorted(pros)  # make spids appear in order
    return pros


def get_reactants(stoich, multi=True):
    """
    Sometimes a stoichiometry has species that are not reactants, whose
    stoichiometry coefficients are zero.
    """
    return get_substrates(stoich, multi) + get_products(stoich, multi)


def get_merged_stoich(*stoichs):
    """
    """
    spids = sorted(set(butil.flatten([stoich.keys() for stoich in stoichs])))
    stoich_m = type(stoichs[0]).fromkeys(spids, 0)
    for stoich in stoichs:
        for spid, stoichcoef in stoich.items():
            stoich_m[spid] += stoichcoef
    for spid, stoichcoef in stoich_m.items():
        if stoichcoef == 0:
            del stoich_m[spid]
    return stoich_m


def reverse_stoich(stoich):
    """
    
    """
    stoich_b = type(stoich).fromkeys(stoich.keys(), 0)  # b for backward
    for spid, stoichcoef in stoich.items():
        stoich_b[spid] = -1 * stoichcoef
    return stoich_b


def get_all_states(spids):
    """
    """
    states = butil.get_powerset(spids)
    states_sort = sorted([tuple(sorted(s)) for s in states])
    return states_sort


def stoich2eqn(stoich):
    pass

def eqn2stoich(eqn):
    """
    Convert reaction equation (a string) to stoichiometry (a dictionary).
    """
    def unpack(s):
        # an example of s: ' 2 ATP '
        l = filter(None, s.split(' '))
        if len(l) == 1:
            stoichcoef, spid = '1', l[0]
        if len(l) == 2:
            stoichcoef, spid = l
        return spid, stoichcoef
    
    # re: '<?': 0 or 1 '<'; '[-|=]': '-' or '=' 
    subs, pros = re.split('<?[-|=]>', eqn)
    stoich = OD([])
    
    for sub in subs.split('+'):
        subid, stoichcoef = unpack(sub)
        stoich[subid] = '-' + stoichcoef
    for pro in pros.split('+'):
        proid, stoichcoef = unpack(pro)
        stoich[proid] = stoichcoef
        
    return stoich
    
    
def get_ratelaw_MM(rxnid, stoich, reversible=True, 
                   Keq=None, KIcs=None, KIus=None, KAs=None,
                   states_allowed=None, states_add=None, one_step=False):
    """
    Construct rate law expressions for Michaelis-Menten kinetics
    with quasi-equilibrium assumption.  
    
    Input: 
        stoich: a dict mapping from species ids to stoich coefs,
                e.g., {'S':-1, 'P':1, 'I':0}
        Keq: a number, e.g., 1e2
        KIcs: a dict mapping from ids of competitive inhibitors to
              inhibition constants,
              e.g., {'I1':2.5e-2, 'I2':1e-1}
        KIus: a dict mapping from ids of uncompetitive inhibitors to
              inhibition constants,
              e.g., {'I1':2.5e-2, 'I2':1e-1}
        states_allowed: a list of tuples, orders not matter
        one_step: one step binding used by Alex Shestov; if True, assume... 
        
    Output:
        
    """
    # Abbreviations:
    # sub: substrate
    # pro: product
    # s: state
    # f: forward
    # b: backward
    
    subs = get_substrates(stoich, multi=True)
    pros = get_products(stoich, multi=True)
    #state_allsubs = tuple(subs)
    #state_allpros = tuple(pros)
    KMs_sub = ['KM_%s_%s'%(rxnid, sub) for sub in subs]
    KMs_pro = ['KM_%s_%s'%(rxnid, pro) for pro in pros]
    spid2KM = dict(zip(subs+pros, KMs_sub+KMs_pro))

    # make a dictionary, state2term, that maps from binding state 
    # of the enzyme to the corresponding term
    states = [()]
    if one_step:
        states.append(tuple(subs))
        states.append(tuple(pros))
    elif states_allowed:
        states.extend(states_allowed)
    else:
        states.extend(get_all_states(subs)+get_all_states(pros))
    if states_add:
        states.extend(states_add)
    # remove duplicates and double sort
    states = sorted(set([tuple(sorted(s)) for s in states]))  
        
    state2term = OD([((), '1')])
    for state in states:
        term = '*'.join(['(%s/%s)'%(spid, spid2KM[spid]) for spid in state])
        if term == '':
            term = '1'
        term = str(sympy.simplify(term))
        state2term[state] = term
    if one_step:
        state2term[tuple(subs)] = '%s/KM_%s_f'%('*'.join(subs), rxnid)
        state2term[tuple(pros)] = '%s/KM_%s_b'%('*'.join(pros), rxnid)
    
    if KIcs:  # add the competitive-inhibitor bound states
        for spid in KIcs.keys():
            term = '(%s/KIc_%s_%s)'%(spid, rxnid, spid)
            state2term[tuple([spid])] = term
    
    #for (state, term) in state2term.items():
    #    state2term[state] = str(sympy.simplify(term))
    
    Vf, Vb = 'Vf_%s'%rxnid, 'Vb_%s'%rxnid
    if Keq:  # use Haldane's relation to cancel Vb
        # Haldane's relation says at equilibrium:
        # Vf*subs/KMs_sub = Vb*pros/KMs_pro; therefore: 
        # pros/subs = Keq = Vf*KMs_pro/(Vb*KMs_sub), 
        # Vb = Vf*KMs_pro/(KMs_sub*Keq)
        Keq = 'Keq_%s'%rxnid
        if one_step:
            KMs_f = 'KM_%s_f'%rxnid
        else:
            KMs_f = '(' + '*'.join(KMs_sub) + ')'
        numerator = '(%s/%s*(%s-%s/%s))'%(Vf, KMs_f, '*'.join(subs),
                                          '*'.join(pros), Keq)
    else:
        numerator = '(%s*%s - %s*%s)'%(Vf, state2term[tuple(subs)], 
                                       Vb, state2term[tuple(pros)])
    if not reversible:
        numerator = '(%s*%s)'%(Vf, state2term[tuple(subs)])
    denominator = '(' + '+'.join(state2term.values()) + ')'
    ratelaw = '(%s/%s)'%(numerator, denominator)
    
    if KIus:
        for spid in KIus.keys():
            term = '%s/KIu_%s_%s'%(spid, rxnid, spid)
            ratelaw = '%s*(1/(1+%s))'%(ratelaw, term)             
    if KAs:
        for spid in KAs.keys():
            term = '%s/KA_%s_%s'%(spid, rxnid, spid)
            ratelaw = '%s*(%s/(1+%s))'%(ratelaw, term, term)
    
    return ratelaw
    
    
def add_reaction_MM(net, id, Vs, stoich, KMs, reversible=True, 
                    Keq=None, deltaG=None, KIcs=None, KIus=None, KAs=None,
                    one_step=False, duplicate_error=True, 
                    param_optimizable=True, 
                    states_allowed=None, states_add=None):
    """
    Requires the species to be in the network
    """
    rxnid = id
    
    # check if the species are already in the network
    
    # add the parameters
    for direction, paramval in Vs.items():
        if isinstance(paramval, tuple):
            paramval = paramval[0]
        paramid = 'V' + direction + '_' + rxnid
        net.add_parameter(paramid, paramval, is_optimizable=True)
    if Keq or deltaG is not None:
        if deltaG is not None:
            R = 8.314
            T = 310.15  # 273.15+37
            Keq = np.exp(-deltaG * 1e3 / R / T)
        #if isinstance(Keq, tuple):
        #    paramval = Keq[0]
        net.add_parameter('Keq_'+rxnid, Keq, is_optimizable=False)
    for spid, paramval in KMs.items():
        if isinstance(paramval, tuple):
            paramval = paramval[0]
        paramid = 'KM_' + rxnid + '_' + spid 
        net.add_parameter(paramid, paramval, is_optimizable=True)
    if KIcs is not None:
        for spid, paramval in KIcs.items():
            if isinstance(paramval, tuple):
                paramval = paramval[0]
            paramid = 'KIc_' + rxnid + '_' + spid
            net.add_parameter(paramid, paramval, is_optimizable=True)
    if KIus is not None:
        for spid, paramval in KIus.items():
            if isinstance(paramval, tuple):
                paramval = paramval[0]
            paramid = 'KIu_' + rxnid + '_' + spid
            net.add_parameter(paramid, paramval, is_optimizable=True)
    if KAs is not None:
        for spid, paramval in KAs.items():
            if isinstance(paramval, tuple):
                paramval = paramval[0]
            paramid = 'KA_' + rxnid + '_' + spid
            net.add_parameter(paramid, paramval, is_optimizable=True)
 
    
    # get the rate law
    ratelaw = get_ratelaw_MM(rxnid=rxnid, stoich=stoich, reversible=reversible,
                             one_step=one_step, Keq=Keq, 
                             KIcs=KIcs, KIus=KIus, KAs=KAs, 
                             states_allowed=states_allowed, 
                             states_add=states_add)
    # add the reaction
    net.addReaction(rxnid, stoich, kineticLaw=ratelaw)
    net.reactions.get(rxnid).kinetics = 'MichaelisMenten'


def add_reaction_SMM(net, id, Vs, stoich_or_eqn, KMs, reversible=True, 
                     Keq=None, deltaG=None, KIcs=None, KIus=None, KAs=None,
                     duplicate_error=True, param_optimizable=True):
    """
    SMM: standard Michaelis-Menten kinetics, referring to quasi-equilibrium 
         and random-order assumptions
    """
    if isinstance(stoich_or_eqn, 'str'):
        stoich = eqn2stoich(stoich_or_eqn)
    else:
        stoich = stoich_or_eqn

    
def add_transport_MM(net, rxnid, stoich, V, KM, reversible=True):
    """
    Transporter of Michaelis-Menten kinetics with the same V and KM.
    """
    id_V = 'V_' + rxnid
    id_sub = get_substrates(stoich, multi=False)[0]
    id_pro = get_products(stoich, multi=False)[0]
    spid = min([id_sub, id_pro], key=len)  # the shorter one is the spid
    id_KM = 'KM_%s_%s'%(rxnid, spid)
    if reversible:
        ratelaw = '%s/%s*(%s-%s)/(1+%s/%s+%s/%s)'%(id_V, id_KM, id_sub, id_pro, 
                                                   id_sub, id_KM, id_pro, id_KM)
    else:
        raise NotImplementedError
        # there is no id_pro
        #ratelaw = '%s*%s/%s/(1+%s/%s)'%(id_V, id_sub, id_KM, id_sub, id_pro)
    net.add_parameter(id_V, V, is_optimizable=True)
    net.add_parameter(id_KM, KM, is_optimizable=True)
    net.addReaction(rxnid, stoich, kineticLaw=ratelaw)


def get_ratelaw_MA(rxnid, stoich, reversible, Keq):
    subs = get_substrates(stoich, multi=True)
    prods = get_products(stoich, multi=True)
    kf = 'kf_' + rxnid
    kb = 'kb_' + rxnid
    if Keq:
        Keq = 'Keq_' + rxnid
    if not reversible:
        ratelaw = '%s*%s' % (kf, '*'.join(subs))
    else:
        if Keq:
            kb = '(%s/%s)' % (kf, Keq)
        ratelaw = '%s*%s-%s*%s' % (kf, '*'.join(subs), 
                                   kb, '*'.join(prods))
    return ratelaw

    
def add_reaction_MA(net, id, ks, stoich, reversible=True, Keq=None, 
                    duplicate_error=True):
    """
    """
    rxnid = id
    
    # check if the species are already in the network
    for spid in stoich.keys():
        if spid not in net.species.keys():
            raise StandardError("Error when adding reaction %s:\
                                 species %s has not been added yet." 
                                    % (rxnid, spid))
    # add the parameters
    for direction, paramval in ks.items():
        if isinstance(paramval, tuple):
            paramval = paramval[0]
        paramid = 'k' + direction + '_' + rxnid
        net.add_parameter(paramid, paramval, is_optimizable=True)
    if Keq:
        net.add_parameter('Keq_'+rxnid, Keq, is_optimizable=False)
    # get the rate law
    ratelaw = get_ratelaw_MA(rxnid=rxnid, stoich=stoich, 
                             reversible=reversible, Keq=Keq)
    # add the reaction
    net.addReaction(rxnid, stoich, kineticLaw=ratelaw)
    net.reactions.get(rxnid).kinetics = 'MassAction'

def get_ratelaw_MWC():
    pass


def add_reaction_MWC():    
    pass
    

class IrreversibleMultisubstrateMichaelisMenten(Reaction):
    """
    
    """

    def __init__(self, id, stoichiometry, info_inhibitors=None, activators=None):
        """
        Parameters
        ----------
        info_inhibitors, a dict
            example:
            [{'inhibitor': 'I1', 'substrate': 'S1', 'mode': 'competitive'},
             {'inhibitor': 'I1', 'substrate': 'S1', 'mode': 'uncompetitive'},
             {'inhibitor': 'I2', 'substrate': 'S2', 'mode': 'competitive'},
             {'inhibitor': 'I3', 'substrate': 'S1', 'mode': 'competitive'}
            ]
            Let:
                id = 13
                substrates = ['S1', 'S2']
            Then: 
                kineticLaw = 'Vm_13*S1*S2/((Km_13_S1*(1+I1/Kic_13_S1_I1+I3/Kic_13_S1_I3)+
                              S1*(1+I1/Kiu_13_S1_I1))*(Km_13_S2*(1+I2/Kic_13_S2_I2)+S2))'
        activators, a dict
            example: to be implemented.
        """
        str_Vm = 'Vm_' + id
        substrates = get_substrates(stoichiometry)
        kineticLaw_numerator = str_Vm + '*' + '*'.join(substrates)
        kineticLaw_denominator_components = []
        for substrate in substrates:
            str_Km = 'Km_' + id + '_' + substrate
            str_concn = substrate
            if info_inhibitors:  # if the reaction has inhibitors
                for info in info_inhibitors:
                    if isinstance(info, tuple):
                        inhibitor, substrate_inhibited, mode = info
                    else:
                        substrate_inhibited = info['substrate']
                        inhibitor = info['inhibitor']
                        mode = info['mode']
                    if substrate == substrate_inhibited:
                        if mode == 'competitive' or mode == 'c':
                            str_Ki = 'Kic_' + id + '_' + substrate + '_' + inhibitor
                            if not str_Km.endswith(')'):
                                str_Km = str_Km + '*(1+' + inhibitor + '/' +  str_Ki + ')'
                            else:
                                str_Km = str_Km[:-1] + '+' + inhibitor + '/' + str_Ki + ')'
                        else:  # uncompetitive inhibition
                            str_Ki = 'Kiu_' + id + '_' + substrate + '_' + inhibitor
                            if not str_concn.endswith(')'):
                                str_concn = str_concn + '*(1+' + inhibitor + '/' +  str_Ki + ')'
                            else:
                                str_concn = str_concn[:-1] + '+' + inhibitor + '/' + str_Ki + ')'
            else:  # if the reaction has no inhibitors
                pass
            kineticLaw_denominator_components.append('('+str_Km+'+'+str_concn+')')
        kineticLaw_denominator = '*'.join(kineticLaw_denominator_components)
        kineticLaw = kineticLaw_numerator + '/(' + kineticLaw_denominator + ')'
        Reaction.__init__(self, id, stoichiometry, kineticLaw)


class ReversibleMassAction(Reaction):
    """
    """
    def __init__(self, id, stoichiometry):
        """
        """
        str_substrates = '*'.join(['pow('+sp+','+str(abs(stoichiometry[sp]))+')'
                                   for sp in stoichiometry.keys() if stoichiometry[sp]<0])
        str_products = '*'.join(['pow('+sp+','+str(stoichiometry[sp])+')'
                                 for sp in stoichiometry.keys() if stoichiometry[sp]>0])
        kineticLaw = 'k_'+id+'*('+str_substrates+'-'+str_products+'/'+'Ke_'+id+')'
        Reaction.__init__(self, id, stoichiometry, kineticLaw)
        

def add_diseq_variables(net, rxnid2dG0=None, 
                        add_Q=False, add_dG=False, add_g=True, 
                        skip_no_dG0=True, T=25):
    """
    Add disequilibrium variables
    
    dG = deltaG
    dG0 = deltaG0
    g = dG/RT
    g0 = dG0/RT
    
    Input:
        T: temperature in celsius degrees (~25 for room temperature and ~37 for
            human body)

    """
    net = net.copy()
    
    for rxnid, rxn in net.reactions.items():
        subs = get_substrates(rxn.stoichiometry)
        pros = get_products(rxn.stoichiometry)
        str_sub = '*'.join(subs)
        str_pro = '*'.join(pros)
        str_Q = '(%s)/(%s)' % (str_pro, str_sub)
        id_Q = 'Q_' + rxnid
        
        if add_Q:
            net.add_parameter(id_Q)
            net.add_assignment_rule(id_Q, str_Q)
        
        RT = 8.315 * (T + 273.15) / 1000
        ## get deltaG0 (G0)
        if rxnid2dG0 and rxnid in rxnid2dG0:
            dG0 = rxnid2dG0[rxnid]
        elif hasattr(rxn, 'deltaG0'):
            dG0 = rxn.deltaG0
        elif hasattr(rxn, 'dG0'):
            dG0 = rxn.dG0
        else:
            if skip_no_dG0:
                continue
            else:
                raise Exception("deltaG0 for rxn %s undefined." % rxnid)
            
        if add_dG:
            id_dG = 'dG_' + rxnid
            if add_Q:
                str_dG = '%.3f + %.3f * log(%s)' % (dG0, RT, id_Q)
            else:
                str_dG = '%.3f + %.3f * log(%s)' % (dG0, RT, str_Q)
            net.add_parameter(id_dG)
            net.add_assignment_rule(id_dG, str_dG)
        
        if add_g:
            id_g = 'g_' + rxnid
            if add_Q:
                str_g = '%.3f + log(%s)' % (dG0/RT, id_Q)
            else:
                str_g = '%.3f + log(%s)' % (dG0/RT, str_Q)
            net.add_parameter(id_g)
            net.add_assignment_rule(id_g, str_g)
            
    return net
