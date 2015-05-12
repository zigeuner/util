"""
Some utility functions for the SloppyCell Network instances.
"""

from __future__ import division
import re

import numpy as np
from SloppyCell.ReactionNetworks import *
from SloppyCell import ExprManip as expr

from util import butil
from util.sloppycell import rxnutil
reload(butil)
reload(rxnutil)


def print_net(net):
    """
    """
    print "Species:\n\t",\
          [(sp.id, sp.value) for sp in net.species]
    print "Optimizable Parameters:\n\t",\
          [(v.id, v.value) for v in net.optimizableVars]
    print "Non-optimizable Parameters:\n\t",\
          [(p.id, p.value) for p in net.parameters if not p.is_optimizable]
    print "Assignment Rules:\n\t",\
          net.assignmentRules.items()
    print "Rate Rules:\n\t",\
          net.rateRules.items()
    

def clean(net, optvarids=None):
    """
    Return a new network with:
      1. variables whose values are determined by boundary conditions
         set as constant;
      2. only parameters who ids are in the argument optvarids set as 
         optimizable variables.
    """
    net = net.copy()
    for var in net.variables:
        ## Set variables determined by boundary conditions as constant.
        try:
            if var.is_boundary_condition:
                net.set_var_constant(var.id, True)
        except AttributeError:
            pass
        ## Set optimizable variables.
        try:
            if var.id in optvarids:
                net.set_var_optimizable(var.id, True)
            else:
                net.set_var_optimizable(var.id, False)
        except TypeError:
            pass
    return net


def copy_piecemeal(net, netid=None):
    """
    Copy the following attributes:
        compartments
        species
        parameters
        assignedVars, assignmentRules
        constantVars
        optimizableVars
        dynamicVars
        algebraicVars, algebraicRules
        rateRules
        Events
    """
    if netid is None:
        netid = net.id
    net2 = Network(id=netid)
    # variables include Compartments, Species and Parameters
    net2.variables = net.variables.copy()
    net2.reactions = net.reactions.copy()
    net2.assignmentRules = net.assignmentRules.copy()
    net2.algebraicRules = net.algebraicRules.copy()
    net2.rateRules = net.rateRules.copy()
    net2.events = net.events.copy()
    # _makeCrossReferences takes care of at least the following attributes:
    # assignedVars, constantVars, optimizableVars, dynamicVars, algebraicVars
    net2._makeCrossReferences()
    return net2


def add_fluxes(net):
    """
    """
    net = net.copy()  # keeping the original copy intact
    net.fluxVars = KeyedList()
    for rxn in net.reactions:
        fluxid = 'J_' + rxn.id
        net.add_parameter(fluxid, is_constant=False, is_optimizable=False)
        net.add_assignment_rule(fluxid, rxn.kineticLaw)
        net.fluxVars.set(fluxid, net.assignedVars.get(fluxid))
    return net


def same_param(net, paramvals, optimizable=True):
    """
    paramvals can be a mapping, hence the complication
    Input:
        optimizable: if True, then compare paramvals to the optimizablevars
                     if False, then to the parameters
        
    """
    if optimizable:
        varids = net.optimizableVars.keys()
        varvals = [var.value for var in net.optimizableVars]
    else:
        varids = net.parameters.keys()
        varvals = [var.value for var in net.parameters]
    try:  # a mapping 
        paramvals = butil.get_values(paramvals, varids)
    except AttributeError:  # a sequence
        paramvals = list(paramvals)
    if paramvals == varvals:
        return True
    else:
        return False


def update_net(net, paramvals=None, dynvarvals=None, time=np.inf):
    """
    libmca.update_net
    """
    if paramvals is not None:
        net.update_optimizable_vars(paramvals)
    if dynvarvals is not None:
        # this will update the assigned variables of the network as well
        net.updateVariablesFromDynamicVars(dynvarvals, time=time)
        
        
def get_drugged_net(net, paramid, fold, netid=None):
    """
    Return a new net, with kinetic laws of reactions and 
    rules (assignment and algebraic) changed so that the given parameter
    has a fold change given by "fold", to mimic the drug perturbation.
    """
    if netid is None:
        netid = '%s_drugged_%s_%s'%(net.id, paramid, str(fold))
    # net_drugged
    net_d = copy_piecemeal(net, netid=netid)
    
    sub_string = lambda string: re.sub('(?<!\w)%s(?!\w)'%paramid, 
                                       '(%s*%s)'%(paramid, str(fold)), string)
    for rxn in net_d.reactions:
        rxn.kineticLaw = sub_string(rxn.kineticLaw)
    for ruleid, rulestr in net_d.assignmentRules.items():
        net_d.assignmentRules.set(ruleid, sub_string(rulestr))
    for ruleid, rulestr in net_d.algebraicRules.items():
        net_d.algebraicRules.set(ruleid, sub_string(rulestr))
    return net_d


def reset_variable_ids(net, netid=None, func=None, oldid2newid=None):
    """
    
    Input:
        One and only one of func and oldid2newid should be given. 
        func: a function that takes in any string of old ids and
              converts it to a string of new ids
        
    """
    net2 = net.copy()
    
    if netid:
        net2.set_id(netid)
    
    ## make func to be used later 
    if oldid2newid:
        def func(oldstr):
            newstr = oldstr
            for oldid, newid in oldid2newid.items():
                newstr.replace(oldid, newid)
            return newstr
    
    ## update variable ids    
    for var in net2.variables:
        var.id = func(var.id)
        
    ## update stoichiometries
    for rxn in net2.reactions:
        setattr(rxn, 'stoichiometry', 
                butil.change_items(rxn.stoichiometry, func_key=func))
    
    ## update assignment rules
    net2.assignmentRules = butil.change_items(net2.assignmentRules, 
                                                func_key=func, func_value=func)
    
    ## update rate rules
    net2.rateRules = butil.change_items(net2.rateRules, 
                                          func_key=func, func_value=func)
    
    return net2

"""
class StoichiometryMatrix(np.matrix):

    def __new__(cls, matrix0, dynvarids=None, rxnids=None):
        obj = np.matrix(matrix0).view(cls)
        obj.dynvarids = dynvarids
        obj.rxnids = rxnids
        obj.nspp = len(dynvarids)
        obj.nrxns = len(rxnids)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.dynvarids = getattr(obj, 'dynvarids', None)
        self.rxnids = getattr(obj, 'rxnids', None)
        self.nspp = getattr(obj, 'nspp', None)
        self.nrxns = getattr(obj, 'nrxns', None)
"""    
"""
    def get_conserved_pools_old(self, eps=1e-10):
        ""
        integer basis of left null space of stoich mat
        Vh: Hermitian of V
        ""
        U, S, Vh = np.linalg.svd(self.transpose())
        # nullspace is a matrix whose ROWs are basis vectors of
        # nullspace of stoichmat.transpose()
        nullspace = Vh[self.nspp - np.linalg.matrix_rank(self), :]
        pools = []
        for vec in nullspace:
            intvec = self.__class__.integerize(vec)
            pools.append(dict(zip(self.dynvarids, intvec.flat)))
        return pools
"""

"""
    @staticmethod
    def integerize(floatvec, k=3, eps=1e-6):
        ""
        A lame solution to the problem detailed here:
        http://stackoverflow.com/questions/14407579/how-to-get-the-integer-eigenvectors-of-a-numpy-matrix

        positive
        ""
        n = floatvec.size
        for i in range(1, n * k**2):
            floatvec2 = floatvec * np.sqrt(i)
            floatvec3 = np.round(floatvec2)
            if np.linalg.norm(floatvec2 - floatvec3) < eps:
                intvec = np.array(floatvec3, dtype='int')
                intvec_pos = np.abs(intvec)  # pos for positive
                if not np.array_equal(intvec_pos, intvec) and \
                   not np.array_equal(intvec_pos, -intvec):
                    raise Exception('Float vector not all positive or negative.')
                return intvec_pos
        raise Exception('Float vector failed to be integerized.')
    

    @staticmethod
    def pool2poolstr(pool):
        ""
        It logically belongs to here.
        ""
        trantab = string.maketrans(':,', '*+')
        poolstr = pool.__str__().translate(trantab, "\'{} ")
        return poolstr


    @staticmethod
    def get_remove_varids(pools):
        ""
        return a list of varids whose values are to be removed
        from ...
        ""
        def get_remove_varids_trial(pools):
            rmvarids = []
            for pool in pools:
                varids = [varid for varid in pool.keys() if pool[varid]!=0]
                rmvarids.append(random.choice(varids))
            return rmvarids
            
        rmvarids = get_remove_varids_trial(pools)
        # check if there are identical elements in rmvarids;
        # repeat the random ensemble if there are, as only one dynvar
        # is removed for each conservation constraint
        while len(rmvarids) != len(set(rmvarids)):
            rmvarids = get_remove_varids_trial(pools)
        return rmvarids


def get_pools(net, recalculate=False):
    if hasattr(net, 'pools') and not recalculate:
        return net.pools
    
    if hasattr(net, 'stoichmat'):
        stoichmat = net.stoichmat
    else:
        stoichmat = get_stoich_mat(net)
    pools = stoichmat.get_conserved_pools()
    net.pools = pools
    return pools


def get_remove_varids(net, recalculate=False):

    if hasattr(net, 'rmvarids') and not recalculate:
        return net.rmvarids
    
    if hasattr(net, 'pools'):
        pools = net.pools
    else:
        pools = get_pools(net, recalculate=recalculate)
    rmvarids = StoichiometryMatrix.get_remove_varids(pools)
    net.rmvarids = rmvarids
    return rmvarids
"""


"""
def get_funcs(net):
    dynvarids = net.dynamicVars.keys()
    constvarids = net.constantVars.keys()
    paramids = net.parameters.keys()
    constvarvals = [var.value for var in net.constantVars]
    paramvals = [p.value for p in net.parameters.values()]

    stoichmat = get_stoich_mat(net)
    pools = stoichmat.get_conserved_pools()

    def get_maps(concns, params):
        paramid2val = dict(zip(constvarids, constvarvals)) 
        paramid2val.update(dict(zip(paramids, params)))
        dynvarid2val = dict(zip(dynvarids, concns))
        return dynvarid2val, paramid2val

    def ratefunc_rxns(concns, params=paramvals):
        ""
        do I need to use this func?
        ""
        dynvarid2val, paramid2val = get_maps(concns, params)
        kineticlaws = [expr.sub_for_vars(rxn.kineticLaw, paramid2val)
                       for rxn in net.reactions]
        rates_rxns = [eval(kl, dynvarid2val) for kl in kineticlaws]
        return rates_rxns
    
    def ratefunc_spp(concns, params=paramvals):
        rates_rxns = ratefunc_rxns(concns, params)
        rates_rxns = np.array(rates_rxns).reshape((len(rates_rxns),1))
        rates_spp = (stoichmat * rates_rxns).flatten().tolist()
        return rates_spp

    picks = StoichiometryMatrix.pick_from_pools(pools)
    def fsolvefunc(concns, params=paramvals):
        rates_spp = ratefunc_spp(concns, params)
        rates_spp_reduced = [rates_spp[i] for i in range(len(rates_spp))
                             if stoichmat.dynvarids[i] not in picks]
        
        # Adding conserved pool relationships
        poolstrs = []
        for pool in pools:
            poolsize = 0
            for varid, multiplicity in pool.items():
                poolsize += net.variables.get(varid).value * multiplicity
            poolstr = StoichiometryMatrix.pool2poolstr(pool)
            poolstrs.append(poolstr + '-' + str(poolsize))

        dynvarid2val, paramid2val = get_maps(concns, params)
        delta_poolsizes = [eval(ps, dynvarid2val) for ps in poolstrs]
        return rates_spp_reduced + delta_poolsizes
    
    return ratefunc_rxns, ratefunc_spp, fsolvefunc
"""

def get_toy_net(kinetics='MA', paramid2val=None):
    """
    make a network with a simple irreversible conversion of S to P 
    with mass action kinetics
    """
    net = Network('net')
    net.add_compartment('cell')
    net.add_species('S', 'cell', 1)
    net.add_species('P', 'cell', 0)
    if kinetics == 'MA':
        net.add_parameter('k', 1)
        net.addReaction('R', {'S':-1, 'P':1}, 'k*S')
    if kinetics == 'MM':
        net.add_parameter('V', 1)
        net.add_parameter('K', 1)
        net.addReaction('R', {'S':-1, 'P':1}, 'V*S/(K+S)')
    return net


def smod2net(filepath):
    """
    """
    
    def format(input):
        # remove whitespace
        # add a preceding underscore if the species id starts with a number
        def f(spid):
            spid = spid.replace(' ', '')
            if spid[0] in '0123456789':
                return '_' + spid
            else:
                return spid
        if isinstance(input, str):
            return f(input)
        if isinstance(input, list):
            return map(f, input)
        
    fh = open(filepath)
    string = ''.join(fh.readlines())
    fh.close()
    
    mod, spp, rxns = filter(None, string.split('@'))
    
    # re: preceded by '=', followed by some whitespace and '(', nongreedy (?) 
    netid = re.search('(?<=\=).*?(?=\s*\()', mod).group()
    # re: preceded by '(', followed by ')'
    note = re.search('(?<=\().*(?=\))', mod).group()
    
    net = Network(netid)
    net.note = note
    net.add_compartment('Cell')
    
    for sp in filter(None, spp.split('\n'))[1:]:
        if '(' in sp:
            spid, concn = sp.split('(')[0].split(':')
            if '1' in sp.split('(')[1]:  # constancy flag: constant=1
                is_constant = True
            else:
                is_constant = False
        else:
            spid, concn = sp.split(':')
            is_constant = False
        spid = format(spid)
        net.add_species(spid, 'Cell', float(concn), is_constant=is_constant)
        
    for rxn in filter(None, rxns.split('\n'))[1:]:
        rxnid = format(rxn.split(':')[0])
        # re: get what comes after the first ':'
        eqn, rate, params = re.split('^.*?:', rxn)[1].split(';')
        
        """
        ## modifiers
        
        activators = []
        inhibitors = []
        if '(' in eqn:
            # re: between '(' and ')'
            for modifiers in re.search('(?<=\().*(?=\))', eqn).group().split('|'):
                if modifiers.replace(' ', '')[0] == '+':
                    activators.extend(format(modifiers.split(':')[1].split(',')))
                if modifiers.replace(' ', '')[0] == '-':
                    inhibitors.extend(format(modifiers.split(':')[1].split(',')))
            eqn = eqn.split('(')[0]
        """
        
        if '<->' in eqn:
            rev = True
        else:
            rev = False
        
        ratelaw = rate.split('v=')[1]
        
        if ratelaw == 'SMM':
            rxnutil.add_reaction_SMM(net, rxnid, stoich_or_eqn=eqn)
    
    return net