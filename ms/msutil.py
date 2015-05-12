"""
"""

from collections import OrderedDict as OD
import copy

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from SloppyCell.ReactionNetworks import *

from util import butil, plotutil
reload(butil)
reload(plotutil)


class MSData(pd.DataFrame):
    """
    MSData:
        Rows are indexed by time;
        Columns are multiindexed by a subset of (genotype, condition, species, 
            nlabel, replicate) with the same order.
    
    Rough personal notes of pandas DataFrame terminology:
        index: an index object with (levels of) labels and name(s), 
               can be for row or columns; or row index of a dataframe
        label: entries of an index object;
               or the index of levels in a multiindex (confusing)
        column: column index of a dataframe
        level: one-dimensional index which, when cartesian-multiplied with
               other levels, forms multiindex
        name(s): can be thought as the index(s) of indices
        values: data in a series, dataframe, etc. (others being metadata)
        
    pandas dataframe: 
        []: column-first; should be avoided for setting values, view vs. copy
        get: column only; can provide default, works for multiindex
        attribute accessing: column only
        query: row only
        loc, iloc, ix: all row-first
        xs: row-first
        sortlevel: row-first
        groupby: row-first
        
    Finally column is chosen to be multiindexed and row for indexing time:
        1. conceptually more natural and seems more conforming to what
            dataframes are about (a bunch of series);
        2. [] is the most common operation: 
            http://stackoverflow.com/questions/22059089/
            multiindexing-rows-vs-columns-in-pandas-dataframe/22085192
        3. query is not that important.
    """
    
    def sort(self, inplace=False):
        return self.sortlevel(axis=1, inplace=inplace)
        
    
    def copy(self):
        return MSData(copy.deepcopy(self))
        
    
    def get_index_values(self, idxname):
        """
        """
        if idxname == 'time':
            return self.index.tolist()
        else:
            levidx = self.columns.names.index(idxname)
            return self.columns.levels[levidx].tolist()
    
    
    def get_xsection(self, drop_level=True, **idxname2vals):
        """
        This method gets a cross section.
            - a customization of xs for MSData
            - a generalization of xs to allow a list of values) 
        
        Input:
            idxname2vals: idx names ('time' or level names such as 'species') 
                          mapped to some of their values/labels 
                          (scalar or lists)
        """
        df = self.copy()
        for idxname, idxvals in idxname2vals.items():
            if idxname == 'time':
                df = df.loc[idxvals]
            else:
                if hasattr(idxvals, '__iter__'):
                    cidxs = butil.get_indices(df.columns.get_level_values(idxname), 
                                              idxvals)
                    df = df[cidxs]
                else:
                    df = df.xs(idxvals, level=idxname, axis=1,
                               drop_level=drop_level)
        return MSData(df)
    
    
    def rename_level(self, levname, levvals, inplace=False):
        """
        
        Reference:
        https://github.com/pydata/pandas/issues/4160 (THM: not really working)
        
        Input: 
            levvals_new: a seq or a map or a func
        """
        levidx = self.columns.names.index(levname) 
        
        # get f
        if hasattr(levvals, 'get'):  # a map
            f = lambda key: key[:levidx]+(levvals[key[levidx]],)+key[levidx+1:]
        elif hasattr(levvals, 'func_name'):  # a func
            f = lambda key: key[:levidx]+(levvals(key[levidx]),)+key[levidx+1:]
        else:  # a seq
            f = lambda key: key[:levidx]+(levvals[levidx],)+key[levidx+1:]
        
        cindex = pd.MultiIndex.from_tuples([f(key) for key in 
                                            self.columns.get_values()], 
                                            names=self.columns.names)
        if inplace:
            self.columns = cindex
        else:
            df = MSData(self.copy())
            df.columns = cindex
            return df
        
    
    def check_level(self, levidx, levname):
        """
        """
        idx2word = {0:'first', 1:'second', 2:'third', 3:'fourth', 4:'fifth',
                    -1:'last'}
        if self.columns.names[levidx] != levname:
            raise Exception("The %s level is NOT %s."%(idx2word[levidx],
                                                       levname))
        
    
    def apply_func(self, func, idxname):
        """
        Apply a function at a certain level. 
        """
        if idxname == 'time':
            # return a series
            return self.apply(func, axis=0)
        else:
            levidx = self.columns.names.index(idxname)
            gb = self.groupby(level=self.columns.names[:levidx], axis=1)
            return MSData(gb.agg(func))
        
        
    def append_level(self, levname, levval):
        """
        
        Reference:
        http://stackoverflow.com/questions/14744068/
        prepend-a-level-to-a-pandas-multiindex
        """
        df = self.T
        df[levname] = levval
        df.set_index(levname, append=True, inplace=True)
        return MSData(df.T)
    
    
    def add_summed_nlabels(self):
        """
        """
        levidx = self.columns.names.index('nlabel')
        gb = self.groupby(level=self.columns.names[:levidx], axis=1)
        def f(group):
            gb_group = group.groupby(level='replicate', axis=1)
            return gb_group.agg(np.sum)
        df_sum = pd.concat(dict([(key,f(group)) for key, group in gb]), axis=1)
        df_sum.columns.set_names(self.columns.names[:levidx]+['replicate'], 
                                 inplace=True) 
        df_sum = MSData(df_sum).append_level('nlabel', 'Sum')
        df_sum = df_sum.reorder_levels(self.columns.names, axis=1)
        
        df_all = pd.concat([self, df_sum], axis=1)
        return MSData(df_all)
        
        
    
    def get_stats(self, ret_one_df=True, ret_one_col=False):
        """
        Returns two MSData instances, one for means and the other for 
        standard errors (sem) of replicates.
        """
        # 'replicate' level should come last
        self.check_level(-1, 'replicate')
        df_mean = MSData(self.apply_func(np.mean, 'replicate'))
        df_sem = MSData(self.apply_func(np.std, 'replicate')/\
                        np.sqrt(len(self.get_index_values('replicate'))))
        if ret_one_df:
            df_mean = df_mean.append_level('stats', 'mean')
            df_sem = df_sem.append_level('stats', 'sem')
            df_stats = MSData(pd.concat([df_mean, df_sem], axis=1)).sort()
            if ret_one_col:
                gb = df_stats.groupby(level=range(df_stats.columns.nlevels-1), 
                                      axis=1)
                df_stats = MSData(gb.apply(lambda g: g.apply(tuple, axis=1)))
            return df_stats
        else:
            return df_mean, df_sem
    
    
    def plot_alllabels(self, plot_sum=False, group_condition=True,
                       spids_order=None, folderpath='.'):
        """
        This function plots the time series of different nlabels, to help
        inspect which label corresponds to C13 signal and others correspond to 
        noise. 
        """
        if plot_sum:
            df = self.add_summed_nlabels()
        else:
            df = self.copy()
        levidx = df.columns.names.index('nlabel')
        if group_condition:
            gb = df.groupby(level='species', axis=1)
        else:
            gb = df.groupby(level=df.columns.names[:levidx], axis=1)

        if spids_order is not None:
            items = [('%02d_%s'%(idx+1,spid), gb.get_group(spid)) 
                     for idx, spid in enumerate(spids_order)]
        else:
            items = [(spid, group) for spid, group in gb]
            
        for spid, group in items:
            g_mean, g_sem = MSData(group).get_stats(ret_one_df=False)  
            trajs_x = df.get_index_values('time')
            trajs_y = np.array(g_mean).T
            trajs_err = np.array(g_sem).T
            legends = g_mean.columns.droplevel('species').get_values()
            if isinstance(spid, tuple):
                filepath = '%s/%s.pdf'%(folderpath.rstrip('/'), '_'.join(spid))
            else:
                filepath = '%s/%s.pdf'%(folderpath.rstrip('/'), spid)
            colors = ['b','g','r','c','m','y','k','#C0C0C0','#FFA500','#FFC0CB']
            colors = colors[:len(trajs_y)/2]
            fmts = ['-'+c for c in colors] + ['--'+c for c in colors]
            plotutil.plot(trajs_x, trajs_y, trajs_err=trajs_err,
                          xmin=-1, xmax=16, ytickformat='sci', fmts=fmts,
                          legends=legends, figtitle=spid.split('_')[-1], filepath=filepath)
            
    
    def test_uniformity(self):
        """
        This method returns the pvalue of the uniformity test.
        """
        if 'Sum' not in self.get_index_values('nlabel'):
            df = self.add_summed_nlabels()
        else:
            df = self.copy()
        df_stats = df.get_stats(ret_one_df=True).get_xsection(nlabel='Sum')
        levidx = df_stats.columns.names.index('stats')
        gb = df_stats.groupby(level=df_stats.columns.names[:levidx], axis=1)
        def get_pvalue(group):
            means, errs = group.iloc[:,0], group.iloc[:,1]
            mu = np.sum(means/errs**2)/np.sum(1/errs**2)
            p = np.prod(1-sp.stats.norm.cdf(np.abs(means-mu)/errs))
            return p
        return MSData(gb.apply(get_pvalue))
    
    
    def get_major_nlabel(self, ret_data=True):
        """
        Get the nlabel that represents the C13 signal.
        """
        nlabels = [n for n in self.get_index_values('nlabel')[1:] if n!='Sum'] 
        df_C13 = self.get_xsection(nlabel=nlabels, drop_level=False)
        levidx = df_C13.columns.names.index('nlabel')
        gb = df_C13.groupby(level=df_C13.columns.names[:levidx], axis=1)
        key2maxnlabel = gb.apply(lambda group: group.groupby(level='nlabel', 
                                 axis=1).mean().mean().idxmax()) 
        if ret_data:
            df_C12 = self.get_xsection(nlabel=0, drop_level=False)
            gs = [g.xs(key2maxnlabel[key], level='nlabel', axis=1, 
                       drop_level=False) for key, g in gb]
            df_C13_major_nlabel = pd.concat(gs, axis=1)
            df = MSData(pd.concat([df_C12, df_C13_major_nlabel], axis=1))
            nlabelmap = lambda nlabel: (['C12']+
                ['C13']*(len(self.get_index_values('nlabel'))-1))[nlabel]
            df.rename_level('nlabel', nlabelmap, inplace=True)
            return df 
        else:
            return key2maxnlabel
        
        
    def get_C12_C13(self):
        """
        """
        gb = self.groupby(level='species', axis=1)
        def drop_masses(group):
            mass_max = max(group.columns.get_level_values('nlabel'))
            return group.drop(range(1,mass_max), axis=1, level='nlabel')
        return MSData(gb.apply(drop_masses))
        
        
    def to_expt(self, cond2netid, C12=False, fix_sf=True):
        """
        
        """
        ## 
        if C12:
            df = self.copy()
        else:
            df = self.get_xsection(nlabel='C13')
          
        ## get data  
        # KFP
        if len(cond2netid) == 1:
            expt = Experiment('expt')
            data = None  # ...
            expt.set_data(data)
            if fix_sf:
                expt.set_fixed_sf(dict.fromkeys(data.keys(), 1))
            return expt
        
        # rKFP
        else:
            cond2suffix = butil.change_values(cond2netid, 
                                              lambda netid: netid[-2:])
            
            def f(tu):
                if len(tu)==2 or tu[2]=='C13':
                    C = '_l'
                else:
                    C = '_u'
                return tu[1].lower()+C+cond2suffix[tu[0]]

            gb = df.groupby(level='condition', axis=1)
            expts = OD()
            for cond, df_cond in gb:
                statdf_cond = MSData(df_cond).get_stats(ret_one_df=True, 
                                                        ret_one_col=True)
                statdf_cond.columns = [f(item) for item in statdf_cond.columns.tolist()]
                data_cond = butil.change_keys(statdf_cond.to_dict(), 
                                              lambda key: ''.join(key))
                for spid, data_cond_sp in data_cond.items():
                    if '_l' in spid:
                        data_cond_sp[0] = (0, data_cond_sp[0][1])
                    # regularize the data, discard sems that are too small
                    errs = [stat[1] for stat in data_cond_sp.values()]
                    err_mean = np.sort(errs)[2:].mean()  # usually min is 0, so discard the smallest two
                    for t, stat in data_cond_sp.items():
                        mean, err = stat
                        if err == 0 or err < 0.1*err_mean:
                            data_cond_sp[t] = (mean, err_mean)

                        if not stat[0]>0 and not stat[1]>0:  # np.nan
                            data_cond_sp.pop(t)

                exptid = 'expt'+cond2suffix[cond]
                expt_cond = Experiment(exptid)
                expt_cond.set_data({cond2netid[cond]:data_cond})
                if fix_sf:
                    expt_cond.set_fixed_sf(dict.fromkeys(data_cond.keys(), 1))
                expts[exptid] = expt_cond
            return expts
        
    
    
"""    
       #def get_expt(self, netid, attrdict=None, metabolites=None, times=None,
       #          exptid=None, check_numbers=True, fix_sf=True):

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


class MassSpecDatum(object):
    """
    """
    
    attrnames = ('celltype', 'genotype', 'condition', 'metabolite', 'label',
                 'position', 'time', 'replicate')
    # attrnames_timeseries = ('celltype', 'genotype', 'condition', 
    #                        'metabolite', 'label', 'position')
    
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
    
    """
    def get_attrvals(self, attrnames=None):
        
        #Return a tuple of attribute values 
        #in the order of (given) attribute names.
        
        if attrnames is None:
            attrnames = MassSpecDatum.attrnames
        return tuple(libtype.get_values(self.__dict__, attrnames))

    
    def get_attrdict(self, attrnames=None):

        if attrnames is None:
            attrnames = MassSpecDatum.attrnames
        return libtype.get_submapping(self.__dict__, attrnames)
    """
    
    
class MassSpecData(list):
    """
    """
    
    def __add__(self, other):
        return MassSpecData(self +other)
        
    
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
                                         while there are %d replicates.' % \
                                        (nval, str(attrdict_time, nrep)))
            # sem: sd divided by sqrt(n)
            ts[time] = (np.mean(values),
                        np.std(values) / np.sqrt(len(values)))
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
                mean, sem = np.mean(values), np.std(values) / np.sqrt(len(values))
                # if zero, choose a heuristic characteristic scale
                if sem == 0:
                    sem = np.mean(msdata.get_values({'metabolite':mb})) * 0.1
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
            sem_sum = np.sqrt(sem ** 2 + sem2 ** 2)
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
        means, sems = dat[:, 0], dat[:, 1]
      
        def sumsq(y):
            return np.sum(((means - y) / sems) ** 2)
        
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
        ys, yerrs = dat[:, 0], dat[:, 1]
        ax.errorbar(range(len(times)), ys, yerrs)
        if plot_fit:
            y_fit = self.fit_horizontal_line(times_fit)
            ax.plot([0, len(times) - 1], [y_fit, y_fit])
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels(times)
        ax.set_xlim(-0.5, len(times) - 0.5)
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
