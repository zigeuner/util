from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict as OD

class Enzyme(object):
    def __init__(self, name=None, ecnum=None, stoich=None, ratelaw=None):
        self.name = name
        self.ecnum = ecnum
        self.stoich = stoich
        self.ratelaw = ratelaw
        self.KMs = []
        self.KIs = []
        self.turnover_numbers = []
        self.molecular_weights = []
        self.specific_activities = []
        
    def fill_parameters(self, brendaenz, paramclass):
        """
        brendaenz: an instance of brenda.parser.Enzyme
        paramclass: 'KMs', 'KIs', 'turnover_numbers', 
                    'molecular_weights', 'specific_activities'
        """
        if paramclass == 'KMs':
            key = u'KM_VALUE'
        if paramclass == 'KIs':
            key = u'KI_VALUE'
        if paramclass == 'turnover_numbers':
            key = u'TURNOVER_NUMBER'
        if paramclass == 'molecular_weights':
            key = u'MOLECULAR_WEIGHT'
        if paramclass == 'specific_activities':
            key = u'SPECIFIC_ACTIVITY'
        ecnum = str(brendaenz.ec_number)
        for brendaparam in brendaenz.entries[key]:
            try:
                if str(brendaparam.msg).startswith('-'):  # negative
                    continue
                elif 'e-' in str(brendaparam.msg):
                    paramvals = [float(str(brendaparam.msg))]
                elif '-' in str(brendaparam.msg):
                    paramvals = [float(paramval) for paramval in
                                 str(brendaparam.msg).split('-')]
                else:
                    paramvals = [float(str(brendaparam.msg))]
            except ValueError:
                print str(brendaparam.msg)
                continue
            metabolite = str(brendaparam.information)
            spp = [str(brendaenz.organisms[spidx].name)
                   for spidx in brendaparam.organisms]
            if brendaparam.comment is None or\
                'mutant' not in brendaparam.comment.msg:
                mutant = False
            else:
                mutant = True
            references = brendaparam.references
            for sp in spp:
                for paramval in paramvals:
                    param = Parameter(ecnum=ecnum, paramclass=paramclass,
                                      metabolite=metabolite, species=sp, 
                                      value=paramval, references=references,
                                      mutant=mutant)
                    getattr(self, paramclass).append(param)
    
    def fill_all_parameters(self, brendaenz):
        paramclasses = ['KMs', 'KIs', 'turnover_numbers', 'molecular_weights',
                        'specific_activities']
        for paramclass in paramclasses:
            self.fill_parameters(brendaenz, paramclass=paramclass)
        self.calculate_turnover_numbers()
        
    
    def get_metabolite_count(self, paramclass, cutoff=1):
        """
        params: 'KMs', 'KIs'
        """
        metabolites = [p.metabolite for p in getattr(self, paramclass)]
        if 'more' in metabolites:
            metabolites.remove('more')
        metabolite_count = Counter(metabolites).items()
        # remove metabolites with few counts
        metabolite_count = [item for item in metabolite_count
                            if item[1]>cutoff]
        # sort metabolite by counts
        metabolite_count = sorted(metabolite_count,
                                  key=lambda item: item[1], reverse=True)
        metabolite2count = OD(metabolite_count)
        return metabolite2count
    
    def get_parameter_ensemble(self, paramclass, inattrdict=None, outattrnames=['value']):
        if inattrdict is None:
            params = getattr(self, paramclass)
        else:
            params = []
            for param in getattr(self, paramclass):
                try:
                    # if all the attribute conditions are satisfied
                    if all([getattr(param, inattrname)==inattrval or
                            getattr(param, inattrname) in inattrval
                            for inattrname, inattrval in inattrdict.items()]):
                        params.append(param)
                except TypeError:
                    pass
        ens = [tuple([getattr(param, outattrname) for outattrname in outattrnames])
               for param in params]
        # if only output parameter value
        if outattrnames == ['value']:
            ens = [item[0] for item in ens]
        return ens

    def calculate_turnover_numbers(self):
        """
        [specific activity]: umol/(min mg)
        [turnover number]: umol/(sec umol)
        1/min = 1/60 1/sec
        1/mg = wt/10**3 1/umol
        """
        for specific_activity in self.specific_activities:
            species = specific_activity.species
            wts = self.get_parameter_ensemble(paramclass='molecular_weights',
                                              inattrdict={'species':species})
            if wts:
                meanwt = np.mean(wts)
                val_specific_activity = specific_activity.value
                val_turnover_number = val_specific_activity/60*meanwt/10**3
                mutant = specific_activity.mutant
                turnover_number = Parameter(ecnum=self.ecnum, mutant=mutant,
                                    paramclass='turnover_numbers', 
                                    value=val_turnover_number, species=species)
                getattr(self, 'turnover_numbers').append(turnover_number)

    def plot_hist(self, paramclass, inattrdict=None, log=True, bins=10, xlim=None, title='',
                  filepath='', cutoff=5):
        vals = self.get_parameter_ensemble(paramclass=paramclass, 
                                           inattrdict=inattrdict)
        if len(vals) < cutoff:
            return
        logvals = np.log10(vals)
        
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.hist(logvals, bins=bins)
        ax.set_xlabel('$log_{10}$ (%s)' % paramclass)
        if not title:
            if inattrs is None:
                title = self.name
            else:
                title = self.name + '\n' + str(inattrs)[1:-1].replace("'", "")
        ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        plt.savefig(filepath, dpi=300)
        
    def save_data(self, paramclass, inattrdict=None, header=None, filepath=''):
        """
        append data to a file
            header: description of the data; e.g., EC number, enzyme name, metabolite name 
            data: a list
        """
        vals = self.get_parameter_ensemble(paramclass=paramclass, 
                                           inattrdict=inattrdict)
        if header is None:
            header = '%s\t%s\t%s' % (self.ecnum, paramclass, str(inattrdict)[1:-1])
        fh = open(filepath, 'a')
        fh.write(header+'\n')
        fh.write(str(vals)+'\n')
        fh.close()
    
    
class Parameter(object):
    """
    Michaelis constant KM or inhibition constant KI
    """
    def __init__(self, ecnum=None, paramclass=None, metabolite=None,
                 species=None, value=None, references=None, mutant=None):
        self.ecnum = ecnum
        self.paramclass = paramclass
        self.metabolite = metabolite
        self.species = species
        self.references = references
        self.value = value
        self.mutant = mutant

def inspect_mechanism(brendaenz, count=True):
    str_mechanism = brendaenz.entries[u'REACTION'][0].comment.msg
    strs_mechanism = str_mechanism.encode('ascii', 'ignore').split('; ')[1:]
    if count:
        count_random = len([s for s in strs_mechanism if 'random' in s])
        count_ordered = len([s for s in strs_mechanism if 'ordered' in s])
        return {'random': count_random, 'ordered': count_ordered}
    else:
        return strs_mechanism
    