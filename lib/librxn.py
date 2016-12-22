from collections import OrderedDict as OD
from itertools import chain, combinations
from SloppyCell.ReactionNetworks.Reactions import Reaction
import libtype

def get_substrates(stoich):
    return [spid for spid, stoichcoef in stoich.items() if stoichcoef<0]

def get_products(stoich):
    return [spid for spid, stoichcoef in stoich.items() if stoichcoef>0]

def get_reactants(stoich):
    """
    Sometimes a stoichiometry has species that are not reactants, whose
    stoichiometry coefficients are zero.
    """
    return get_substrates(stoich) + get_products(stoich)

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


def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s,r) for r in range(len(s)+1)))
    
def get_MM_ratelaw(rxnid, stoich, reversible=True, Keq=None, KIcs=None,
                KIus=None, one_step=False, states_allowed=None):
    """
    
    """
    if Keq:
        Keq = 'Keq_' + rxnid
    Vf = 'Vf_' + rxnid
    Vb = 'Vb_' + rxnid  
    subs = get_substrates(stoich)
    prods = get_products(stoich)
    KMsubs = ['KM_'+rxnid+'_'+sub for sub in subs]
    KMprods = ['KM_'+rxnid+'_'+prod for prod in prods]
    # substrate/product concentrations normalized by Michaelis constants
    subs_norm = ['('+subs[i]+'/'+KMsubs[i]+')' for i in range(len(subs))]
    prods_norm = ['('+prods[i]+'/'+KMprods[i]+')' for i in range(len(prods))]
    ## make a dictionary, state2term, that maps from binding state 
    ## of the enzyme to the corresponding term
    states = [frozenset(s) for s in powerset(subs)] +\
             [frozenset(s) for s in powerset(prods)]
    terms = ['*'.join(s) for s in powerset(subs_norm)] +\
            ['*'.join(s) for s in powerset(prods_norm)]
    state2term = OD(zip(states, terms))
    state2term[frozenset([])] = '1'
    state_subs = frozenset(subs)
    state_prods = frozenset(prods)
    # not random order, only certain bound states are allowed
    if states_allowed:
        # add the unbound state
        if () not in states_allowed: 
            states_allowed.insert(0, ())
        states_allowed = [frozenset(state) for state in states_allowed]
        state2term = libtype.get_submapping(state2term, states_allowed)
    # add the competitive-inhibitor bound states
    if KIcs:
        for spid in KIcs.keys():
            state = frozenset([spid])
            term = '(%s/%s)' % (spid, 'KIc_'+rxnid+'_'+spid)
            state2term[state] = term
    # add the uncompetitive-inhibitor bound states
    if KIus:
        for spid in KIus.keys():
            state = frozenset([spid]).union(state_subs)
            term = state2term[state_subs] +\
                   '*(%s/%s)' % (spid, 'KIu_'+rxnid+'_'+spid)
            state2term[state] = term
    
    ## construct the numerator
    if Keq:
        # Haldane's relation: at equilibrium Vf*s/KMs = Vb*p/KMp
        # Therefore p/s = Keq = Vf*KMp/(Vb*KMs), Vb = Vf*(KMp/(KMs*Keq))
        Vb = '(%s*%s/(%s*%s))' % (Vf, '*'.join(KMsubs), '*'.join(KMprods), Keq)
    numerator = '(%s*%s - %s*%s)' % (Vf, state2term[state_subs], 
                                     Vb, state2term[state_prods])
    if not reversible:
        numerator = '(%s*%s)' % (Vf, state2term[state_subs]) 
    ## construct the denominator
    denominator = '(' + '+'.join(state2term.values()) + ')'
    
    if one_step:
        pass
    ratelaw = '%s/%s' % (numerator, denominator)
    return ratelaw
    
    
def add_MM_reaction(net, rxnid, Vs, stoich, KMs, reversible=True, KIcs=None, KIus=None, Keq=None,
                    one_step=False, duplicate_error=True, param_optimizable=True, states_allowed=None):
    """
    Requires the species to be in the network
    """
    # check if the species are already in the network
    
    # add the parameters
    for direction, paramval in Vs.items():
        if isinstance(paramval, tuple):
            paramval = paramval[0]
        paramid = 'V' + direction + '_' + rxnid
        net.add_parameter(paramid, paramval, is_optimizable=True)
    if Keq:
        if isinstance(Keq, tuple):
            paramval = Keq[0]
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
    
    # get the rate law
    ratelaw = get_MM_ratelaw(rxnid=rxnid, stoich=stoich, reversible=reversible,
                            Keq=Keq, KIcs=KIcs, KIus=KIus, one_step=one_step, 
                            states_allowed=states_allowed)
    # add the reaction
    net.addReaction(rxnid, stoich, kineticLaw=ratelaw)


def get_MA_ratelaw(rxnid, stoich, reversible, Keq):
    subs = get_substrates(stoich)
    prods = get_products(stoich)
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

    
def add_MA_reaction(net, rxnid, ks, stoich, reversible=True, Keq=None, duplicate_error=True):
    """
    """
    # check if the species are already in the network
    for spid in stoich.keys():
        if spid not in net.species.keys():
            raise StandardException("Error when adding reaction %s:\
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
    ratelaw = get_MA_ratelaw(rxnid=rxnid, stoich=stoich, 
                             reversible=reversible, Keq=Keq)
    # add the reaction
    net.addReaction(rxnid, stoich, kineticLaw=ratelaw)
    