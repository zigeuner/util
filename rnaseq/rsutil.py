"""
From http://www.ncbi.nlm.nih.gov/books/NBK47533/:

Study: A study is a set of experiments and has an overall goal.
Experiment: An experiment is a consistent set of laboratory operations 
            on input material with an expected result.
Sample: An experiment targets one or more samples. Results are expressed 
        in terms of individual samples or bundles of samples as defined by
        the experiment.
Run: Results are called runs. Runs comprise the data gathered for a sample
     or sample bundle and refer to a defining experiment.
Submission: A submission is a package of metadata and/or data objects and 
            a directive for what to do with those objects.


From http://www.ncbi.nlm.nih.gov/books/NBK56913/:

Accession  Accession Name         Definition
SRA        submission accession   The submission accession represents a virtual
                                  container that holds the objects represented 
                                  by the other five accessions and is used to 
                                  track the submission in the archive.
SRP        study accession        A Study is an object that contains the project
                                  metadata describing a sequencing study or project.
SRX        experiment accession   An Experiment is an object that contains the 
                                  metadata describing the library, platform selection,
                                  and processing parameters involved in a particular 
                                  sequencing experiment.
SRR        run accession          A Run is an object that contains actual sequencing 
                                  data for a particular sequencing experiment. 
                                  Experiments may contain many Runs depending on the 
                                  number of sequencing instrument runs that were needed.
SRS        sample accession       A Sample is an object that contains the metadata 
                                  describing the physical sample upon which a sequencing
                                  experiment was performed.
SRZ        analysis accession     An analysis is an object that contains a sequence data 
                                  analysis BAM file and the metadata describing the sequence 
                                  analysis.

Ontology:
SRP: {SRX}, a study can have multiple experiments
SRX: {SRR}, an experiment can have multiple runs
SRR: {SRS}, a run can have multiple samples, but assumed to be only one sample here; 
            a run corresponds to a dataset
"""

from __future__ import division
from collections import OrderedDict as OD
from collections import Counter
import cPickle
import copy
import re
import xml.etree.ElementTree as ET

import xlrd
import scipy as sp
import scipy.stats as stat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from util2 import butil
reload(butil)


class Run(object):
    """
    runid: id for run (SRR...)
    sampleid: id for sample (SRS...)
    indid: id for individual
    lineid: id for genetic line
    studyid: id for study/project (SRP...)
    """
    def __init__(self, runid=None, sampleid=None, indid=None, lineid=None, 
                 tissue=None, time=None, condition=None, exptid=None, 
                 studyid=None, studytext=None, sampletext=None, rep=None,
                 genotype=None, readcounts=None, otherattrs=None):
        self.runid = runid
        self.sampleid = sampleid
        self.indid = indid
        self.lineid = lineid
        self.tissue = tissue
        self.time = time
        self.condition = condition
        self.exptid = exptid
        self.studyid = studyid
        self.studytext = studytext
        self.sampletext = sampletext
        self.rep = rep
        self.genotype = genotype
        self.readcounts = readcounts
        self.otherattrs = otherattrs


class Data(np.ndarray):
    def __new__(cls, dat=None, ridxs=None, cidxs=None, 
                geneids=None, runs=None):
        """
        """
        ## get _dat, the data part of the object
        _dat = copy.deepcopy(dat)  # copy only data, not attributes
        if _dat is None:
            _dat = [[]]
        else:
            try:
                _ = _dat[0,0]
            # TypeError: not an ndarray (sub)type
            # IndexError: 1D array/list, for DataGene
            except (TypeError, IndexError):  
                _dat = np.array(_dat)
            if ridxs is not None:
                _dat = _dat[ridxs, :]
            if cidxs is not None:
                _dat = _dat[:, cidxs]

        ## get geneids
        if geneids is None:
            if not hasattr(dat, 'geneids'):
                geneids = []
            else:
                geneids = dat.geneids
        if geneids != [] and ridxs is not None:
            geneids = list(np.array(geneids)[ridxs])

        ## get samples
        if runs is None:
            if not hasattr(dat, 'runs'):
                runs = []
            else:
                runs = dat.runs
        if runs != [] and cidxs is not None:
            runs = list(np.array(runs)[cidxs])

        obj = np.asarray(_dat, dtype=np.float64).view(cls)
        obj.geneids = geneids
        obj.runs = runs
        return obj


    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.__dict__ = copy.deepcopy(obj.__dict__)


    def compute_basic_attrs(self):
        """
        Basic attributes:
            studyids
            tissues
            studyids_uniq_sort
            tissues_uniq_sort
            colors
            hatchs
            studyid2color
            tissue2hatch
            #nums_nonzero
        """
        studyids = [s.studyid for s in self.runs]
        tissues = [s.tissue for s in self.runs]
        studyids_uniq_sort = OD(sorted(Counter(studyids).items(), 
                                       key=lambda item:item[1], 
                                       reverse=True)).keys()
        tissues_uniq_sort = OD(sorted(Counter(tissues).items(), 
                                      key=lambda item:item[1], 
                                      reverse=True)).keys()

        colors = ['b', 'r', '#006400', '#00FFFF', '#FFFF00', 
                  'm', '#FFA500', '#7FFF00']
        hatchs = [None, '-', '|', '/', '\\', '.', '*', '+']
        studyid2color = OD(zip(studyids_uniq_sort, colors))
        tissue2hatch = OD(zip(tissues_uniq_sort, hatchs))

        self.studyids = studyids
        self.tissues = tissues
        self.studyids_uniq_sort = studyids_uniq_sort
        self.tissues_uniq_sort = tissues_uniq_sort
        self.colors = colors
        self.hatchs = hatchs
        self.studyid2color = studyid2color
        self.tissue2hatch = tissue2hatch


    def deepcopy(self):
        """
        Problem: 

        >>> hasattr(data, 'geneids')
        True
        >>> data2 = copy.deepcopy(data)
        >>> hasattr(data2, 'geneids')
        False
        """
        data = copy.deepcopy(self)
        data.__dict__ = copy.deepcopy(self.__dict__)
        return data


    def save(self, filepath):
        dat = (self, self.geneids, self.runs) 
        fh = open(filepath, 'w')
        cPickle.dump(dat, fh)
        fh.close()

        
    @staticmethod
    def load(filepath):
        fh = open(filepath)
        dat, geneids, runs = cPickle.load(fh)
        fh.close()
        return Data(dat=dat, geneids=geneids, runs=runs)


    def get_runattrvals(self, runattrid):
        return [getattr(run, runattrid) for run in self.runs]


    def get_subdata(self, ridxs=None, cidxs=None, geneids=None, p=None, 
                    raise_error=False, runattrs=None, runattrs_del=None, 
                    numrandrows=None):
        """
        Input:
            ridxs:
            cidxs:
            geneids:
            p:
            runattrs: all of them have to be satisfied
            runattrs_del: all of them have to be satisfied 
                          (if any, iterate it)
        """
        data = self.deepcopy()
        if ridxs is not None or cidxs is not None:
            data = Data(data, ridxs=ridxs, cidxs=cidxs)
        if geneids:
            ridxs = [data.geneids.index(geneid) for geneid in geneids
                     if geneid in data.geneids]
            if raise_error:
                if len(ridxs) == 0:
                    raise StandardError("Data has no given geneids.")
            data = Data(data, ridxs=ridxs)
        if p:
            nums = data.get_nums_expressing_runs()
            ridxs = np.where(np.array(nums) > data.shape[1]*p)[0]
            data = Data(data, ridxs=ridxs)
        if runattrs:
            cidxs = []
            for cidx, run in enumerate(data.runs):
                bools = []
                for k, v in runattrs.items():
                    if isinstance(v, list):
                        bools.append(getattr(run, k) in v)
                    else:
                        bools.append(getattr(run, k) == v)
                if all(bools):
                    cidxs.append(cidx)
            data = Data(data, cidxs=cidxs)
        if runattrs_del:
            cidxs = []
            for cidx, run in enumerate(data.runs):
                bools = []
                for k, v in runattrs_del.items():
                    if isinstance(v, list):
                        bools.append(getattr(run, k) not in v)
                    else:
                        bools.append(getattr(run, k) != v)
                if any(bools):
                    cidxs.append(cidx)
            data = Data(data, cidxs=cidxs)

        if numrandrows:
            ridxs = np.random.choice(xrange(len(data.geneids)), numrandrows,
                                     replace=False)
            data = Data(data, ridxs=ridxs)

        # convert to a DataGene instance if applicable
        if data.shape[0] == 1 or len(data.shape) == 1:
            data = DataGene(dat=data.flatten(), geneids=data.geneids, 
                            runs=data.runs)
        return data


    def sort(self, col=True, runattrid='runid'):
        """
        By default, sort by runid.
        """
        if col:
            runattrvals = [getattr(run, runattrid) for run in self.runs]
            items = zip(range(len(runattrvals)), runattrvals)
            items_sort = sorted(items, key=lambda item: item[1])
            cidxs_sort = [item[0] for item in items_sort]
            ridxs_sort = None
        else:
            items = zip(range(len(self.geneids)), self.geneids)
            items_sort = sorted(items, key=lambda item: item[1])
            ridxs_sort = [item[0] for item in items_sort]
            cidxs_sort = None
        return Data(self, ridxs=ridxs_sort, cidxs=cidxs_sort)

    
    def get_nums_expressing_runs(self):
        if hasattr(self, 'nums'):
            return self.nums
        bool_vec = np.vectorize(bool)
        nums = []  # number of expressing runs
        for i in range(self.shape[0]):
            nums.append(int(np.sum(bool_vec(self[i,:]))))
        self.nums = nums
        return nums

        
    def plot_hist_nums_expressing_runs(self, filepath=''):
        nums = self.get_nums_expressing_runs()
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.hist(nums, normed=True, bins=self.shape[1], histtype='stepfilled', 
                color='r', edgecolor='r')
        ax.set_xlim(0, self.shape[1])
        ax.set_xlabel('Number of Expressing Runs')
        plt.savefig(filepath, dpi=300)
        plt.close()


    def get_distance_mat(self, col=True, square=False):
        """
        Calculate pairwise distance matrix using scipy.spatial.distance.pdist
        """
        if col:
            distmat = pdist(self.transpose(), metric='correlation')
        else:
            distmat = pdist(self, metric='correlation')
        if square:
            distmat = squareform(distmat)
        return distmat


    def plot_distance_mat(self, col=True, run2label=None, filepath=''):
        distmat = self.get_distance_mat(col=col, square=True)
        fig = plt.figure(figsize=(15, 15), dpi=300)
        ax = fig.add_subplot(111)
        cax = ax.matshow(distmat, interpolation='nearest')
        fig.colorbar(cax)
        if run2label:
            labels = [run2label(run) for run in self.runs]
            ax.set_xticks(range(distmat.shape[0]))
            ax.set_yticks(range(distmat.shape[0]))
            ax.set_xticklabels(labels, rotation='vertical', fontsize=3)
            ax.set_yticklabels(labels, fontsize=3)
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
        plt.savefig(filepath, dpi=300)
        plt.close()


    def plot_clustering(self, link='average', run2label=None, run2color=None, 
                        run2marker=None, run2markersize=None, labelsize=5, 
                        figsize=(16,12), figtitle='', filepath=''):
        """
        Distinct colors: http://phrogz.net/css/distinct-colors.html
        """
        if run2label:
            labels_all = [run2label(run) for run in self.runs]
        else:
            labels_all = range(1, len(self.runs)+1)
        if run2color:
            colors_all_old = [run2color(run) for run in self.runs]
        else:
            colors_all_old = ['k'] * len(self.runs)
        if run2marker:
            markers_all_old = [run2marker(run) for run in self.runs]
        else:
            markers_all_old = ['o'] * len(self.runs)
        if run2markersize:
            msizes_all_old = [run2markersize(run) for run in self.runs]
        else:
            msizes_all_old = [500/len(self.runs)] * len(self.runs)

        distmat = self.get_distance_mat(col=True, square=False)
        linkagemat = linkage(distmat, link)

        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111)
        out = dendrogram(linkagemat, color_threshold=0, labels=labels_all,
                         leaf_font_size=labelsize)

        """
        colors = ['#420000', '#821300', '#d55d1c', '#e2c95d', '#4c4000', 
                  '#2e9502', '#004f00', '#001b00', '#00f056', '#00edff',
                  '#005860', '#001e23', '#00acff', '#174361', '#0072ff', 
                  '#000524', '#000495', '#5400e2', '#ff83ff', '#2e0025', 
                  '#fb00a9', '#a70064', '#570026', '#ff0027']
        lineids_uniq_sort = sorted(list(set([run.lineid for run in self.runs])))
        lineid2color = OD(zip(lineids_uniq_sort, colors))
        time2size = {12:1, 16:1.5, 20:2, 24:2.5, 30:3, 36:3.5}
        colors_all_old = [lineid2color[run.lineid] for run in self.runs]
        """
        idxs_new= out['leaves']
        colors_all_new = [colors_all_old[idx] for idx in idxs_new]
        markers_all_new = [markers_all_old[idx] for idx in idxs_new]
        msizes_all_new = [msizes_all_old[idx] for idx in idxs_new]
        
        for idx, label in enumerate(ax.get_xmajorticklabels()):
            label.set_color(colors_all_new[idx])

        for idx, x in enumerate(ax.get_xticks()):
            ax.plot([x], [0], marker=markers_all_new[idx], 
                    markerfacecolor=colors_all_new[idx],
                    markeredgecolor=colors_all_new[idx], 
                    markersize=msizes_all_new[idx])

        ax.set_frame_on(False)
        ax.get_yaxis().tick_left()
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, ymax = ax.get_yaxis().get_view_interval()
        ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='k', 
                                 linewidth=2))
        
        ax.set_ylim(bottom=-ymax/50)
        ax.set_title(figtitle)
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.savefig(filepath, dpi=300)
        plt.close()
        

    def plot_checkerboard(self, cidxs=None, run2label=None, filepath=''):
        data = Data(dat=self, cidxs=cidxs)
        n = data.shape[1]
        if run2label:
            labels = [run2label(run) for run in np.array(self.runs)[cidxs]]
        else:
            labels = range(1, n+1)

        fig = plt.figure(dpi=300, figsize=(n, n))
        for i, j in np.ndindex((n, n)):
            data_i = data[:, i]
            data_j = data[:, j]
                
            ax = fig.add_subplot(n, n, i*n+j+1)
            ax.scatter(data_i, data_j, s=1, marker='o', facecolor='r', lw=0)
            print labels[i], labels[j], data_i[:10]-data_j[:10]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_xlabel(labels[j], fontsize=8)
                ax.xaxis.set_label_position('top')
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            if j == n-1:
                ax.set_ylabel(labels[i], fontsize=8)
                ax.yaxis.set_label_position('right')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(filepath, dpi=300)
        plt.close()

    
    def get_outliers(self, threshold_sd=1.5):
        """
        From Ding Zehong:
        I referred a paper (Oldham MC, et al. (2008) Functional organization of
        the transcriptome in human brain. Nat Neurosci 11:1271-1282.) in which 
        outliers were considered as: samples with an average inter-array
        correlations (IACS) less than two-fold (here I used 1.5-fold) standard
        deviation below the mean IAC, and tried to remove outlier in our case.
        
        mIRC: mean inter-run correlation
        """
        distmat = self.get_distance_mat(square=True)
        corrcoefmat = 1 - distmat
        mIRCs = (np.sum(corrcoefmat, axis=0)-1) / (corrcoefmat.shape[0]-1)
        #print corrcoefmat
        #print mIRCs
        idxs = np.where(mIRCs < np.mean(mIRCs) - threshold_sd*np.std(mIRCs))[0]
        return idxs


    def pca(self):
        pass


    def plot_pca(self):
        pass

    
    def plot_pie(self, study_or_tissue):
        """
        """
        if study_or_tissue == 'study':
            data = self.studyids
            keys = self.studyids_uniq_sort
        if study_or_tissue == 'tissue':
            data = self.tissues
            keys = self.tissues_uniq_sort
        
        counts = butil.get_values(Counter(data), keys)
        pctfunc = lambda pct: str(int(len(data)*pct/100))

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.pie(counts, labels=keys, autopct=pctfunc, 
               pctdistance=0.85)
        plt.subplots_adjust(top=0.9)
        plt.suptitle('%s Distribution of the %d Runs'\
                     %(study_or_tissue.capitalize(), len(data)), 
                     fontsize=16)
        plt.axis('equal')
        plt.savefig('plot_pie_%s.png'%study_or_tissue, dpi=300)
        plt.close()


    def plot_pie_nested(self, tissue_in_study=True):
        if tissue_in_study:
            counter = Counter(zip(self.studyids, self.tissues))
        else:
            counter = Counter(zip(self.tissues, self.studyids))
        # counter_items_sort1: sorted by counts
        counter_items_sort1 = sorted(counter.items(), key=lambda item:item[1], 
                             reverse=True)
        # counter_items_sort2: sorted by both studies and counts
        counter_items_sort2 = []
        for studyid in self.studyids_uniq_sort:
            for item in counter_items_sort1:
                if item[0][0] == studyid:
                    counter_items_sort2.append(item)
        counter = OD(counter_items_sort2)

        colors_patch = [studyid2color[item[0][0]] for item in counter.items()]
        hatchs_patch = [tissue2hatch[item[0][1]] for item in counter.items()]

        fig = plt.figure(dpi=300)
        gss = gs.GridSpec(1, 3, width_ratios=[1, 0, 0])
        ax = fig.add_subplot(gss[0])
        patchs, texts = ax.pie(counter.values(), labels=counter.values(),
                               colors=colors_patch, startangle=90)[:2]
        for patch, hatch in zip(patchs, hatchs_patch):
            patch.set_hatch(hatch)
        for text in texts:
            text.set_fontsize(8)
        ax.set_xmargin(0.2)
        ax.axis('equal')

        ax = fig.add_subplot(gss[1])
        bars_color = [ax.bar([0], [0], color=c) for c in studyid2color.values()]
        ax.axis('off')
        ax = fig.add_subplot(gss[2])
        bars_hatch = [ax.bar([0], [0], color='w', hatch=h) 
                      for h in tissue2hatch.values()]
        ax.axis('off')
        plt.legend(bars_color+bars_hatch, studyid2color.keys()+tissue2hatch.keys(),
                   fontsize=9, markerscale=0.8, loc='center right')

        #plt.subplots_adjust(hspace=0.1)
        plt.suptitle('Study and Tissue Distribution of the %d Runs'%self.shape[1], 
                     fontsize=12)
        plt.savefig('plot_pie_study_tissue.png', dpi=300)
        plt.close()


    def standardize(self, map_udg, log10=True, threshold_expression=True,
                    xs_new=None, get_f=None):
        """
        In-place update the tissue positions from unified gradient,
        or, output a new Data instance given the new positions 
        (between 0 and 1) and an interpolating function generator.

        Input: 
            map_udg: a dict
            log10:
            threshold_expression: an int between 0 and 15, the number
                                  of expressing sections, used for deciding
                                  whether to keep the gene
            xs_new: a list of elements between 0 and 1
            get_f: 
        """
        data = self.deepcopy()

        for run in data.runs:
            run.tissue = map_udg[run.tissue]

        if xs_new is None:
            return data
        else:
            data_new = []
            geneids_new = []
            xs = data.get_runattrvals('tissue')
            for idx in range(data.shape[0]):
                data_gene = data[idx]
                geneid = data.geneids[idx]
                if threshold_expression is not None:
                    maxexps = dict(sorted(zip(xs, data_gene))).values()
                    num_expressing_secs = np.sum(np.array(maxexps) > 0)
                    if num_expressing_secs < threshold_expression:
                        continue
                func = DataGene.get_fit_func(data_gene, xs, log10=log10, 
                                             get_f=get_f)
                data_new.append(func(xs_new))
                geneids_new.append(geneid)
            return Data(data_new, geneids=geneids_new, runs=self.runs)


def get_corrcoefs(data1, data2, pairs, filepath=''):
    """
    data1 and data2 have to be of same datalength. 
    """
    ccs = []
    for pair in pairs:
        geneid1, geneids2 = tuple(pair)
        try:
            if isinstance(geneids2, str):
                data_gene1 = data1[data1.geneids.index(geneid1)]
                data_gene2 = data2[data2.geneids.index(geneids2)]
                ccs.append(sp.stats.pearsonr(data_gene1, data_gene2)[0])
            if len(geneids2) == 2:
                geneid2a, geneid2b = geneids2
                data_gene1 = data1[data1.geneids.index(geneid1)]
                data_gene2a = data2[data2.geneids.index(geneid2a)]
                data_gene2b = data2[data2.geneids.index(geneid2b)]
                ccs.append([sp.stats.pearsonr(data_gene1, data_gene2a)[0],
                            sp.stats.pearsonr(data_gene1, data_gene2b)[0]])
        except:
            pass
    ccs = np.array(ccs)
    if filepath:
        np.save(filepath, ccs)
    return ccs
        


class DataGene(Data):
    @staticmethod
    def get_fit_func(data_gene, xs, log10=True, get_f=None):
        if log10:
            ys = np.log10(data_gene)
            idxs_nonzero = np.where(ys != -np.inf)[0]
            xs_fit = np.array(xs)[idxs_nonzero]
            ys_fit = ys[idxs_nonzero]
        else:
            xs_fit = xs
            ys_fit = data_gene
        if not get_f:
            get_f = lambda xs, ys: np.polyfit(xs, ys, 3)
        f = get_f(xs_fit, ys_fit)
        if log10:
            func = lambda xs: np.power(10, np.polyval(f, xs))
        else:
            func = lambda xs: np.polyval(f, xs)
        return func


    def plot_dynamics(self, dynvarid, line=True, plot_fit=False, xlabel='', ylabel='',  
                      figtitle='', filepath=''):
        """
        """

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        dynvarvals = self.get_runattrvals(dynvarid)
        medians = []
        for dynvarval in dynvarvals:
            subdata = self.get_subdata(runattrs={dynvarid:dynvarval})
            ax.plot([dynvarval]*len(subdata), subdata, 'o', color='b')
            medians.append(np.median(np.array(subdata)))
        if line:
            ax.plot(dynvarvals, medians, '-', color='r')
        if plot_fit:
            func = DataGene.get_fit_func(self, dynvarvals, log10=True)
            xs_fit = np.linspace(np.min(dynvarvals), np.max(dynvarvals), 100)
            ax.plot(xs_fit, func(xs_fit), '-', color='r')
        # ax.set_xticks(dynvarvals)
        # ax.set_xlim(0.5, 16.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(figtitle)
        plt.savefig(filepath, dpi=300)
        plt.close()
            


    # functions for maximum likelihood estimations
    @staticmethod
    def logpdf(theta, e, idx_tissue, idx_batch):
        """
        Model:
        e_ijk = e_i + t_ij + b_ik + epsilon, epsilon ~ N(0, sigma2)
        theta = (e_i, t_i1, ..., t_i8, b_i1, ..., b_i8, sigma) (18 in total)
        """
        mu = theta[0] + theta[idx_tissue+1] + theta[idx_batch+1+8]
        sigma = theta[-1]
        return stat.norm.logpdf(e, loc=mu, scale=sigma)


    @staticmethod
    def d_logpdf(theta, e, idx_tissue, idx_batch):
        """
        """
        mu = theta[0] + theta[idx_tissue+1] + theta[idx_batch+1+8]
        sigma = theta[-1]
        d_logpdf_mu = (e - mu)/sigma**2
        d_mu_theta = np.zeros(len(theta))
        for idx  in [0, idx_tissue+1, idx_batch+1+8]:
            d_mu_theta[idx] = 1
            d_logpdf_theta = d_logpdf_mu * d_mu_theta  # chain rule
        d_logpdf_sigma = -1/sigma + (e-mu)**2/sigma**3
        d_logpdf_theta[-1] = d_logpdf_sigma
        return d_logpdf_theta

    @staticmethod
    def get_nll(es, idxs_tissue, idxs_batch):
        """
        nll: negative-log-likelihood
        """
        def nll(theta):
            out = 0 
            for i, e in enumerate(es):
                idx_tissue = idxs_tissue[i]
                idx_batch = idxs_batch[i]
                out = out - DataGene.logpdf(theta, e, idx_tissue, idx_batch)
            return out
        return nll

    @staticmethod
    def get_d_nll(es, idxs_tissue, idxs_batch):
        """
        """
        def d_nll(theta):
            out = np.zeros(len(theta))
            for i, e in enumerate(es):
                idx_tissue = idxs_tissue[i]
                idx_batch = idxs_batch[i]
            out = out - DataGene.d_logpdf(theta, e, idx_tissue, idx_batch)
            return out
        return d_nll


    def mle(self, **kwargs_fmin):
        idxs_tissue = [self.tissues_uniq_sort.index(s.tissue) 
                       for s in self.runs]
        idxs_batch = [self.studyids_uniq_sort.index(s.studyid) 
                      for s in self.runs]

        nll = DataGene.get_nll(self, idxs_tissue, idxs_batch)
        d_nll = DataGene.get_d_nll(self, idxs_tissue, idxs_batch)

        # get initial estimates
        theta0 = [sp.mean(self)]
        for tissue in self.tissues_uniq_sort:
            data_tissue = [e for i, e in enumerate(self) 
                           if self.runs[i].tissue == tissue]
            theta0.append(sp.mean(data_tissue) - sp.mean(self))
            
        for batch in self.studyids_uniq_sort:
            data_batch = [e for i, e in enumerate(self)
                          if self.runs[i].studyid == batch]
            theta0.append(sp.mean(data_batch) - sp.mean(self))
        theta0.append(sp.std(self))
        theta0 = sp.array(theta0)
        
        out = sp.optimize.fmin_bfgs(nll, theta0, d_nll, **kwargs_fmin)
        return out

    
    def plot(self, figtitle=''):
        """
        """
        data_hist = [[copy.copy([]) 
                      for j in range(len(self.studyids_uniq_sort))]
                     for i in range(len(self.tissues_uniq_sort))]

        for idx_datum, datum in enumerate(self):
            sample = self.runs[idx_datum]
            i = self.tissues_uniq_sort.index(sample.tissue)
            j = self.studyids_uniq_sort.index(sample.studyid)
            data_hist[i][j].append(datum)
        
        data_hist = np.asarray(data_hist, dtype='object')    
        xmin = np.floor(min(np.log10(self)))
        xmax = np.ceil(max(np.log10(self)))

        fig = plt.figure(figsize=(6, 12), dpi=300)
        nsubplot = len(self.tissues_uniq_sort)
        gss = gs.GridSpec(nsubplot, 2, width_ratios=[0.8, 0.2], wspace=0.2)

        for ridx, data_tissue in enumerate(data_hist):
            ax = fig.add_subplot(gss[ridx, 0], xlim=(xmin, xmax), yticks=[],
                                 ylabel=self.tissues_uniq_sort[ridx])
        
            for cidx, data_tissue_study in enumerate(data_tissue):
                if data_tissue_study:
                    color = self.colors[cidx]  # study
                    n, bins, patches = ax.hist(np.log10(data_tissue_study), 
                                               normed=1, histtype='bar', 
                                               color=color, edgecolor=color, 
                                               stacked=1, fill=1, alpha=0.8)
        
            if ridx != len(data_hist)-1:
                ax.set_xticklabels([])
            else:
                # f: float
                format = lambda f: re.sub('(?<=e).*(?=\d)', '', '%.0e'%f) 
                xticklabels = [format(f) for f in 
                               np.power(10, ax.get_xticks())]
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('mRNA abundance')
    
        ax = fig.add_subplot(gss[:, -1])
        bars_color = [ax.bar([0], [0], color=c) for c in self.colors]
        ax.axis('off')
        """
        ax = fig.add_subplot(gss[:, -1])
        bars_hatch = [ax.bar([0], [0], color='w', hatch=h) for h in hatchs]
        ax.axis('off')
        """
        plt.legend(bars_color, self.studyids_uniq_sort, 
                   fontsize=9, markerscale=0.8, loc='center right')
        plt.subplots_adjust(hspace=0)
        if not figtitle:
            figtitle = self.geneids + '.png'
        plt.savefig('Plots/'+figtitle, dpi=300)
        plt.close()


    def correct(self, **kwargs_fmin):
        out = self.mle(**kwargs_fmin)
        if isinstance(out, tuple):
            batcheffects = out[0][9:17]
        else:
            batcheffects = out[9:17]
        # bef: batcheffect
        studyid2bef = OD(zip(self.studyids_uniq_sort, batcheffects))
        dat = [e-studyid2bef[self.runs[i].studyid]
               for i, e in enumerate(self)]
        return DataGene(dat=dat, geneids=self.geneids, runs=self.runs)


    def plot_mle(self, figtitle='', **kwargs_fmin):
        data_gene2 = self.correct(**kwargs_fmin)

        data_hist = [[copy.copy([]) 
                      for j in range(len(self.studyids_uniq_sort))]
                     for i in range(len(self.tissues_uniq_sort))]
        data_hist2 = copy.deepcopy(data_hist)

        for idx_datum, datum in enumerate(self):
            run = self.runs[idx_datum]
            i = self.tissues_uniq_sort.index(run.tissue)
            j = self.studyids_uniq_sort.index(run.studyid)
            data_hist[i][j].append(datum)
            data_hist2[i][j].append(data_gene2[idx_datum])
        
        data_hist = np.asarray(data_hist, dtype='object')    
        data_hist2 = np.asarray(data_hist2, dtype='object')
        xmin = np.floor(min(np.log10(self)))
        xmax = np.ceil(max(np.log10(self)))

        fig = plt.figure(figsize=(12, 12), dpi=300)
        nsubplot = len(self.tissues_uniq_sort)
        gss = gs.GridSpec(nsubplot, 3, width_ratios=[0.4, 0.4, 0.2], 
                          wspace=0.2)
        
        ## plot the uncorrected histograms
        for ridx, data_tissue in enumerate(data_hist):
            ax = fig.add_subplot(gss[ridx, 0], xlim=(xmin, xmax), yticks=[],
                                 ylabel=self.tissues_uniq_sort[ridx])
        
            for cidx, data_tissue_study in enumerate(data_tissue):
                if data_tissue_study:
                    color = self.colors[cidx]  # study
                    n, bins, patches = ax.hist(np.log10(data_tissue_study), 
                                               normed=1, histtype='bar', 
                                               color=color, edgecolor=color, 
                                               stacked=1, fill=1, alpha=0.8)
            if ridx == 0:
                ax.set_title('Before correcting batch effect')        
            if ridx != len(data_hist)-1:
                ax.set_xticklabels([])
            else:
                # f: float
                format = lambda f: re.sub('(?<=e).*(?=\d)', '', '%.0e'%f) 
                xticklabels = [format(f) for f in 
                               np.power(10, ax.get_xticks())]
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('mRNA abundance')

        ## plot the corrected histograms
        for ridx, data_tissue2 in enumerate(data_hist2):
            ax = fig.add_subplot(gss[ridx, 1], xlim=(xmin, xmax), yticks=[],
                                 ylabel=self.tissues_uniq_sort[ridx])
        
            for cidx, data_tissue_study2 in enumerate(data_tissue2):
                if data_tissue_study2:
                    color = self.colors[cidx]  # study
                    n, bins, patches = ax.hist(np.log10(data_tissue_study2), 
                                               normed=1, histtype='bar', 
                                               color=color, edgecolor=color, 
                                               stacked=1, fill=1, alpha=0.8)
            if ridx == 0:
                ax.set_title('After correcting batch effect')
            if ridx != len(data_hist2)-1:
                ax.set_xticklabels([])
            else:
                # f: float
                format = lambda f: re.sub('(?<=e).*(?=\d)', '', '%.0e'%f) 
                xticklabels = [format(f) for f in 
                               np.power(10, ax.get_xticks())]
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('mRNA abundance')

        ax = fig.add_subplot(gss[:, -1])
        bars_color = [ax.bar([0], [0], color=c) for c in self.colors]
        ax.axis('off')

        plt.legend(bars_color, self.studyids_uniq_sort, 
                   fontsize=9, markerscale=0.8, loc='center right')
        plt.subplots_adjust(hspace=0)
        if not figtitle:
            figtitle = self.geneids + '.png'
        plt.savefig('Plots/'+figtitle, dpi=300)
        plt.close()        
        
        
# functions for parsing xml metadata files
def decompose_1step(ob, pretag=''):
    """
    """
    if not hasattr(ob, '__dict__'):
        raise StandardError("The object is not composite!")
    
    d_simple, d_composite = OD(), OD()
    _dict = ob.__dict__.copy()
    children = _dict.pop('_children')
    attrs = _dict.pop('attrib')
    del _dict['tag']
    d_simple.update([(pretag+k.lower(), v) for (k, v) in 
                     _dict.items()+attrs.items()])
    for child in children:
        key = (pretag+ob.tag+'_'+child.tag).lower()
        if hasattr(child, '__dict__'):
            d_composite[key] = child
        else:
            d_simple[key] = child
    return d_simple, d_composite
        

def decompose(ob, pretag=''):
    """
    """
    d = OD()
    d_simple, d_composite = decompose_1step(ob, pretag=pretag)
    d.update(d_simple)
    ds_composite = d_composite.copy()
    while ds_composite:
        ds_composite_new = OD()
        for key, ob in ds_composite.items():
            d_simple, d_composite = decompose_1step(ob, pretag=key+'_')
            d.update(d_simple)
            ds_composite_new.update(d_composite)
        ds_composite = ds_composite_new
    return d


def clean(d):
    def rm_peat(key):
        key_new = re.sub(r'(.+?_)\1+', r'\1', key)
        while key_new != key:
            key = key_new
            key_new = re.sub(r'(.+?_)\1+', r'\1', key)
        return key_new

    # remove items with values being '\n' or ''
    for key, val in d.items():
        if not val.replace('\n', '').replace(' ', ''):
            del d[key]

    # simplify the keys
    d_new = OD()
    for key, val in d.items():
        key_new = rm_peat(key)
        d_new[key_new] = val
    return d_new


def parse_xml(id_SRA, filetype):
    """
    Input:
        id_SRA: e.g., 'SRA055066'
        filetype:'experiment', 'run', 'sample', or 'submission'
    """
    tree = ET.parse('Data/raw/NCBI_SRA_Metadata_Full_20130901/%s/%s.%s.xml'%\
                    (id_SRA, id_SRA, filetype))
    root = tree.getroot()
    id2d = OD()
    if len(root) > 0:
        for i in range(len(root)):
            d = clean(decompose(root[i], pretag=filetype+'_'))
            if filetype == 'run':
                _id = d['run_accession']
            if filetype == 'sample':
                _id = d['sample_accession']
            if filetype == 'experiment':
                _id = d['experiment_accession']
            id2d[_id] = d
    return id2d


def parse_metadata_file(filepath_metadata):
    """
    Works for Book1.xls from Ding Zehong.
    """
    id2run = OD()
    wb = xlrd.open_workbook(filepath_metadata)
    sheet = wb.sheets()[0]
    for idx_row in range(1, sheet.nrows):
        row = sheet.row(idx_row)
        runid = str(row[0].value)
        studyid = str(row[1].value)
        studytext = str(row[2].value)
        tissue = str(row[4].value)
        run = Run(runid=runid, studyid=studyid, studytext=studytext, 
                  tissue=tissue)
        id2run[runid] = run
    return id2run


def parse_data_sra(id2run):
    """
    Works for edgeR_count_595files_cpm.xls from Ding Zehong.
    """
    dat = []
    geneids = []
    
    fh = open('Data/raw/edgeR_count_595files_cpm.xls')
    runids_all = fh.readline().strip().split('\t')

    runids_common = [runid for runid in id2run.keys() if runid in runids_all]
    id2run = OD(zip(runids_common, butil.get_values(id2run, runids_common)))

    cidxs = [runids_all.index(runid) for runid in id2run.keys()]
    for line in fh:
        elems = line.strip().split('\t')
        geneid = elems[0]
        dat_gene = [float(elem) for elem in elems[1:]]
        dat.append(dat_gene)
        geneids.append(geneid)
    fh.close()
    dat = np.array(dat)[:, cidxs]
    data = Data(dat=dat, geneids=geneids, runs=id2run.values())
    return data


def parse_data_brutnell():
    """
    Works for maizeMerge_count_cpm_75samples.xls from Ding Zehong.
    """
    dat = []
    geneids = []
    
    fh = open('Data/raw/maizeMerge_count_cpm_75samples.xls')
    labels = fh.readline().strip().split('\t')
    runs = []
    for label in labels:
        segment = int(label.split('_')[0][1:])
        # the segment nums go like 1,2,3,4,6,7,...
        if segment > 5:
            segment = segment - 1
        replicate = int(label.split('_')[1][1:])
        runs.append(Run(runid=label, tissue=segment, rep=replicate))
    
    for line in fh:
       elems = line.strip().split('\t') 
       geneid = elems[0]
       dat_gene = [float(elem) for elem in elems[1:]]
       dat.append(dat_gene)
       geneids.append(geneid)
    fh.close()
    data = Data(dat=dat, geneids=geneids, runs=runs)
    return data


def parse_data_brutnell_uniq():
    """
    Works for maize_danforth_merged_uniqReads_cpm_75samples.xls 
    from Ding Zehong.
    """
    dat = []
    geneids = []
    
    fh = open('Data/raw/maize_danforth_merged_uniqReads_cpm_75samples.xls')
    labels = fh.readline().strip().split('\t')
    runs = []
    for label in labels:
        segment = int(label.split('_')[0][1:])
        # the segment nums go like 1,2,3,4,6,7,...
        if segment > 5:
            segment = segment - 1
        replicate = int(label.split('_')[1][1:-4])
        runs.append(Run(runid=label, tissue=segment, rep=replicate))
    
    for line in fh:
       elems = line.strip().split('\t') 
       geneid = elems[0]
       dat_gene = [float(elem) for elem in elems[1:]]
       dat.append(dat_gene)
       geneids.append(geneid)
    fh.close()
    data = Data(dat=dat, geneids=geneids, runs=runs)
    return data


def parse_data_rice_wang():
    """
    Works for Data/raw/lw_rnaseq/rice-rpkm.csv
    """
    dat = []
    geneids = []
    fh = open('Data/raw/lw_rnaseq/rice-rpkm.csv')
    labels = fh.readline().strip().split(',')
    runs = []
    for label in labels[1:-1]:
        tissue = int(label[2:-1])
        runs.append(Run(tissue=tissue)) 

    for line in fh:
        elems = line.strip().split(',')
        geneid = elems[0][1:-1]
        dat_gene = [float(elem) for elem in elems[1:-1]]
        geneids.append(geneid)
        dat.append(dat_gene)
    fh.close()
    data = Data(dat=dat, geneids=geneids, runs=runs)
    return data
    

def parse_data_maize_wang():
    """
    """
    dat = []
    geneids = []
    fh = open('Data/raw/lw_rnaseq/maize-rpkm.csv')
    labels = fh.readline().strip().split(',')
    runs = []
    for label in labels[1:-1]:
        tissue = int(label[2:-1])
        runs.append(Run(tissue=tissue)) 

    for line in fh:
        elems = line.strip().split(',')
        geneid = elems[0][1:-1]
        dat_gene = [float(elem) for elem in elems[1:-1]]
        geneids.append(geneid)
        dat.append(dat_gene)
    fh.close()
    data = Data(dat=dat, geneids=geneids, runs=runs)
    return data
    

def to_geneid(s):
    if s.startswith('GR'):  # maize
        return s.split('_')[0]
    if s.startswith('LOC_'):  # rice
        return s.split('.')[0]

