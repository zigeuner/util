import itertools

from collections import OrderedDict as OD
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *

import libtype
reload(libtype)
  

class MassSpecDatum(object):
    """
    """
    
    attrnames =            ('celltype', 'genotype', 'condition', 
                            'metabolite', 'label', 'position', 
                            'time', 'replicate')
    attrnames_timeseries = ('celltype', 'genotype', 'condition', 
                            'metabolite', 'label', 'position')
    
    def __init__(self, celltype=None, genotype=None, condition=None, 
                 metabolite=None, label=None, position=None, 
                 time=None, replicate=None, value=None):
        self.celltype = celltype
        self.genotype = genotype
        self.condition = condition
        self.metabolite = metabolite
        self.label = label
        self.position = position
        self.time = time
        self.replicate = replicate
        self.value = value
    
    
    def get_attrvec(self, attrnames=None):
        """
        Return a tuple of attribute values (called *attrvec*)
        in the order of (given) attribute names.
        """
        if attrnames is None:
            attrnames = MassSpecDatum.attrnames
        return tuple(libtype.get_values(self.__dict__, attrnames))
    
    
    def get_attrdict(self, attrnames=None):
        """
        """
        if attrnames is None:
            attrnames = MassSpecDatum.attrnames
        return libtype.get_submapping(self.__dict__, attrnames)
    
    
    
class MassSpecData(list):
    """
    """
    
    def __add__(self, other):
        return MassSpecData(list.__add__(self, other))
        
    
    def save(self, filepath='', readable=True):
        """
        Input:
            readable: if True, then saved to the following format:
                attrvec, value
                E.g., 
                (celltype, genotype, condition, metabolite, label, position, time, replicate) value
                ('HCT116', 'wildtype', '5000uM', 'lactate', 'C13', 3, 2, 1) 134253.112
        """
        pass
    
    
    def load(self, filepath=''):
        pass
        
    
    def get_attrvals(self, attrname='time'):
        attrvals = [getattr(datum, attrname) for datum in self]
        return sorted(list(set(attrvals)))
    
    
    def get_sublist(self, attrdict=None, check_numbers=True):
        if attrdict is None:
            return self
        else:
            # attrvals_all: a list of allowed attrvals for all attrnames
            attrvals_all = []
            for attrname in MassSpecDatum.attrnames:
                if attrname in attrdict:
                    if isinstance(attrdict[attrname], list):
                        attrvals_all.append(attrdict[attrname])
                    else:  # make it a list
                        attrvals_all.append([attrdict[attrname]])
                else:
                    attrvals_all.append(self.get_attrvals(attrname))
            
            sublist = MassSpecData()
            for msdatum in self:
                attrvec = msdatum.get_attrvec()                
                if all([attrval in attrvals_all[idx] 
                        for idx, attrval in enumerate(attrvec)]):
                    sublist.append(msdatum)
                    
            if check_numbers:
                if len(sublist) == 0:
                    raise StandardError("Invalid attribute requirements and\
                        values are returned: attrdict=%s" % (str(attrdict)))
                    
            return sublist
    
    
    def get_values(self, attrdict=None, check_numbers=True):
        """
        """
        sublist = self.get_sublist(attrdict=attrdict, 
                                   check_numbers=check_numbers)
        vals = [datum.value for datum in sublist]
        return vals
    
    
    def get_timeseries(self, attrdict=None, orderedtimes=None, 
                       check_numbers=True):
        """
        Input: 
            check_numbers: whether to check if the number of values used
                           to calculate mean and sem is the same as
                           the number of replicates
        """
        if orderedtimes:
            times = orderedtimes
        else:
            times = self.get_attrvals(attrname='time')
        ts = TimeSeries()
        for time in times:
            attrdict_time = dict(attrdict.items() + [('time', time)])
            values = self.get_values(attrdict=attrdict_time)
            if check_numbers:
                nval = len(values)
                nrep = len(self.get_attrvals('replicate'))
                if nval != nrep:
                    raise StandardError('There are %d values corresponding to %s,\
                                         while there are %d replicates.'%\
                                        (nval, str(attrdict_time, nrep)))
            # sem: sd divided by sqrt(n)
            ts[time] = (np.mean(values),
                        np.std(values)/np.sqrt(len(values)))
        return ts
    
    
    def get_summed_timeseries(self, attrdict=None, orderedtimes=None):
        """
        """
        id2ts = self.get_id2ts(attrdict=attrdict, orderedtimes=orderedtimes)
        return reduce(TimeSeries.__add__, id2ts.values())

    
    def get_id2ts(self, attrdict=None, orderedtimes=None, idattrnames=None):
        """
        id: (metabolite, celltype, genotype, condition, label, position)
        Input:
            idattrnames: 
        """
        if attrdict:
            msdata = self.get_sublist(attrdict=attrdict)
        else:
            msdata = self
        
        id2ts = TimeSeries()
        attrvals_all = [msdata.get_attrvals(attrname)
                        for attrname in MassSpecDatum.attrnames_timeseries]

        # from itertools.product doc:
        # "Cartesian product of input iterables.  Equivalent to nested for-loops."
        for attrvec in itertools.product(*attrvals_all):
            attrdict = dict(zip(MassSpecDatum.attrnames_timeseries, attrvec))
            # attrvec may not exist; e.g., 'label':'C12', 'position':3
            try:
                ts = self.get_timeseries(attrdict=attrdict,
                                         orderedtimes=orderedtimes,
                                         check_numbers=True)
            except StandardError:
                continue
            # define idattrnames to make sure id's have the right order
            if idattrnames is None:
                idattrnames = MassSpecDatum.attrnames_timeseries
            idattrvals = tuple(libtype.get_values(attrdict, idattrnames))
            id2ts[idattrvals] = ts
        return id2ts
    
    
    def get_expt(self, netid, attrdict=None, metabolites=None, times=None,
                 exptid=None, check_numbers=True, fix_sf=True):
        """
        Return the SloppyCell Experiment instances
        
        Input: 
            attrdict: celltype, genotype, condition, label, position
              e.g., 
                {'celltype': None,
                 'genotype': None,
                 'condition': '5000uM',
                 'label': 'C13',
                 'position': 6}
            metabolites: a sequence or a mapping (mapping old names to new names)
        """
        if attrdict:
            msdata = self.get_sublist(attrdict)
        else:
            msdata = self
        if metabolites is None:
            metabolites = msdata.get_attrvals('metabolite')
        if hasattr(metabolites, 'items'):
            mbs, mbs_new = metabolites.keys(), metabolites.values()
        else:
            mbs = metabolites
        if times is None:
            times = msdata.get_attrvals('time')
        if exptid is None:
            exptid = 'expt_' + netid
        expt = Experiment(exptid)
        data = {}
        
        for idx, mb in enumerate(mbs):
            data_mb = {}
            for time in times:
                attrdict = {'metabolite': mb,
                            'time': time}
                values = msdata.get_values(attrdict=attrdict)
                if check_numbers:
                    if len(values) != len(msdata.get_attrvals('replicate')):
                        raise StandardError('Number of values different\
                                             from replicates.')
                mean, sem = np.mean(values), np.std(values)/np.sqrt(len(values))
                # if zero, choose a heuristic characteristic scale
                if sem == 0:
                    sem = np.mean(msdata.get_values({'metabolite':mb}))*0.1
                data_mb[time] = (mean, sem)
            try:
                mb_new = mbs_new[idx]
            except:
                mb_new = mb
            data[mb_new] = data_mb
        expt.set_data({netid: data})
        if fix_sf:
            expt.set_fixed_sf(dict.fromkeys(data.keys(), 1))
        return expt
    
    
    """
    def get_sublist2(self, attrdict=None, attrdict_neg=None):
        sublist = MassSpecData()
        for datum in self:
            # check if those in the attrdict are satisfied
            if attrdict is None:
                take_datum = True
            else:
                if_right_attrs = []
                for (attrname, attrval) in attrdict.items():
                    if isinstance(attrval, list):
                        if getattr(datum, attrname) in attrval:
                            if_right_attrs.append(True)
                        else:
                            if_right_attrs.append(False)
                    else:
                        if getattr(datum, attrname) == attrval:
                            if_right_attrs.append(True)
                        else:
                            if_right_attrs.append(False)
                if all(if_right_attrs):
                    take_datum = True
                else:
                    take_datum = False 
            
            # check if those in the attrdict_neg are satisfied
            if attrdict_neg is None:
                take_datum2 = True
            else:
                if_right_attrs2 = []
                for (attrname, attrval) in attrdict_neg.items():
                    if isinstance(attrval, list):
                        if getattr(datum, attrname) not in attrval:
                            if_right_attrs2.append(True)
                        else:
                            if_right_attrs2.append(False)
                    else:
                        if getattr(datum, attrname) != attrval:
                            if_right_attrs2.append(True)
                        else:
                            if_right_attrs2.append(False)
                if all(if_right_attrs2):
                    take_datum2 = True
                else:
                    take_datum2 = False
                    
            # take datum if both are satisfied
            if take_datum and take_datum2:
                sublist.append(datum)
                

            # a more concise code before allowing attrdict(_neg) to have 
            # lists as values
            if (attrdict is None or
                all([getattr(datum, attrname)==attrval 
                     for (attrname, attrval) in attrdict.items()])) and\
               (attrdict_neg is None or
                all([getattr(datum, attrname)!=attrval
                     for (attrname, attrval) in attrdict_neg.items()])):
                sublist.append(datum)

        return sublist
    """
    
        
    """
    @staticmethod
    def get_attrdict(diffattrdict, msdatum=None):

        if msdatum:
            attrdict = msdatum.__dict__.copy()
            _ = attrdict.pop('value')  # remove the 'value' attribute
            attrdict.update(diffattrdict)
        else:
            attrdict = diffattrdict
        # remove items whose values are None
        attrdict = dict(filter(lambda i:i[1] != None, attrdict.items()))
        return attrdict
    """


    """    
    def get_summed_timeseries2(self, attrdict=None, labels=None, positions=None,
                               orderedtimes=None):

        attrdict: celltype, genotype, condition, metabolite

        if labels is None:
            labels = self.get_attrvals('label')
        if positions is None:
            positions = self.get_attrvals('position')
        if orderedtimes is None:
            orderedtimes = self.get_attrvals('time')
        timeseries = OD()
        for time in orderedtimes:
            sums = []
            for replicate in self.get_attrvals('replicate'):
                attrdict_all = dict(attrdict.items() + 
                                    [('time', time),
                                     ('label', labels),
                                     ('position', positions),
                                     ('replicate', replicate)])
                sums.append(np.sum(self.get_values(attrdict_all)))
            timeseries[time] = (np.mean(sums), np.std(sums) / np.sqrt(len(sums)))
        return timeseries
    """    



class TimeSeries(OD):
    """
    """
    
    def __add__(self, other):
        """
        Add two timeseries together:
            means: sum
            sems: root-sum-squares
        """
        if self.keys() != other.keys():
            raise StandardException('The two timeseries have different times.')
        ts = TimeSeries()
        for time in self.keys():
            mean, sem = self.get(time)
            mean2, sem2 = other.get(time)
            mean_sum = mean + mean2
            sem_sum = np.sqrt(sem**2 + sem2**2)
            ts[time] = (mean_sum, sem_sum)
        return ts
    
    
    def fit_horizontal_line(self, times_fit=None):
        """
        """
        if times_fit:
            ts = libtype.get_submapping(self, times_fit)
        else:
            ts = self
            
        dat = np.array(ts.values())
        means, sems = dat[:,0], dat[:,1]
      
        def sumsq(y):
            return np.sum(((means-y)/sems)**2)
        
        y_fit = sp.optimize.leastsq(sumsq, 0)[0][0]
        return y_fit
    
    
    def plot(self, plot_fit=False, times_fit=None, ybottom=None,
             figtitle='', filepath=''):
        """
        Input:
            plot_fit:
            times_fit:
        """
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        times, dat = self.keys(), np.array(self.values())
        ys, yerrs = dat[:,0], dat[:,1]
        ax.errorbar(range(len(times)), ys, yerrs)
        if plot_fit:
            y_fit = self.fit_horizontal_line(times_fit)
            ax.plot([0, len(times)-1], [y_fit, y_fit])
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels(times)
        ax.set_xlim(-0.5, len(times)-0.5)
        if ybottom is not None:
            ax.set_ylim(bottom=ybottom)
        plt.title(figtitle)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
    

def plot_timeseries(id2ts, id2y=None, log10=False, legend=False, markers=None,
                    colors=None, figtitle='', filepath=''):
    """
    Input:
        id2y: a reference level, usually the estimated saturation level
        markers: 
    """
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for id, ts in id2ts.items():
        ys = np.array([val[0] for val in ts.values()])
        yerrs = np.array([val[1] for val in ts.values()])
        if log10:
            yerrs = np.log10(yerrs / ys)
            ys = np.log10(ys)
        if markers:
            marker = markers.pop(0)
        else:
            marker = '-'
        if colors:
            fmt = marker + colors.pop(0)
        else:
            fmt = marker
        ax.errorbar(range(len(ts)), ys, yerrs, fmt=fmt)
        
        # plot horizontal fits
        try:
            y = id2y[id]
            print id, figtitle, y, "\n"
            ax.plot([ts[0], ts[-1]], [y, y], fmt=fmt)
        except:
            pass
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(ts.keys())
    ax.set_xlim(-0.5, len(ts) - 0.5)
    if log10:
        yticks = ax.get_yticks()
        func = np.vectorize(lambda a: format(a, '.1E'))
        ax.set_yticklabels(func(10 ** yticks))
    if legend:
        ax.legend(id2ts.keys(), prop={'size': 9})
    plt.title(figtitle)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    
    
"""
def traj2expt(traj, netid, datvarids=None, exptid=None, CV=0.05, 
              fix_sf=True, add_noise=False):

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
