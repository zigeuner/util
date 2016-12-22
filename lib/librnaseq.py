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

"""

import numpy as np
import cPickle
import copy


class Sample(object):
    def __init__(self, id, studyid=None, studytitle=None, tissue=None, genotype=None, condition=None):
        self.id = id
        self.studyid = studyid
        self.studytitle = studytitle
        self.tissue = tissue
        self.genotype = genotype
        self.condition = condition
        

class Data(np.ndarray):
    def __new__(cls, dat=None, geneids=None, samples=None):
        
        if hasattr(dat, 'geneids') and geneids is None:
            geneids = dat.geneids
        if hasattr(dat, 'samples') and samples is None:
            samples = dat.samples
        if dat is None:
            dat = []
        if geneids is None:
            geneids = []
        if samples is None:
            samples = []
        obj = np.asarray(dat, dtype=np.float64).view(cls)
        obj.geneids = geneids
        obj.samples = samples
        return obj


    def __array__finalize__(self, obj):
        if obj is None: 
            return
        self.geneids = getattr(obj, 'geneids', None)
        self.samples = getattr(obj, 'samples', None)
    

    def slice(self, ridxs):
        geneids = list(np.array(self.geneids)[ridxs])
        data = Data(dat=self[ridxs], geneids=geneids, samples=self.samples)
        return data 
        
    
    def add_line(self, line, geneid):
        shape = self.shape
        if shape == (0,):  # initial state
            dat = line
        else:
            dat = np.vstack((self, line))
        return Data(dat, geneids=self.geneids+[geneid], samples=self.samples)
        """
        self.geneids.append(geneid)
        
        if shape == (0,):  
            x = raw_input('if')
            self = Data([line], self.geneids, self.samples)
            print "here: ", self.shape
        else:
            x = raw_input('else')
            # not working for np.matrix, even when setting refcheck=False
            # it is the reason for subclassing np.ndarray instead of np.matrix
            self.resize((shape[0]+1, shape[1]))
            self[shape[0]] = np.array(line)
            print self.shape
        """ 


    def deepcopy(self):
        """
        Problem: 

        >>> hasattr(data, 'geneids')
        True
        >>> data2 = copy.deepcopy(data)
        >>> hasattr(data2, 'geneids')
        False

        Have to write my own codes for deepcopying.
        """
        dat = copy.copy(self)
        geneids = copy.copy(self.geneids)
        samples = copy.copy(self.samples)
        return Data(dat=dat, geneids=geneids, samples=samples)


    def delete(self, ridxs=None, cidxs=None):
        """
        Input: 
            ridx: row index
            cidx: column index
        """
        if ridxs is not None:
            dat = np.delete(self, ridxs, axis=0)
            geneids = list(np.delete(np.array(self.geneids), ridxs))
            data = Data(dat=dat, geneids=geneids, samples=self.samples)
        if cidxs is not None:
            dat = np.delete(self, cidxs, axis=1)
            samples = list(np.delete(np.array(self.samples), cidxs))
            data = Data(dat=dat, geneids=self.geneids, samples=samples)
        return data


    def clean(self, dicts):
        """
        Input: 
            dicts: dicts of sample attributes for which the data need 
                   to be removed;
                   e.g., [{'studyid':'SRP009313','tissue':'embryo'},
                          {'studyid':'SRP010680','tissue':'root'}]
        """
        data = self.deepcopy()
        for d in dicts:
            idxs = []
            for idx, sample in enumerate(data.samples):
                if all([getattr(sample, k)==v for k, v in d.items()]):
                    idxs.append(idx)
            data = data.delete(cidxs=idxs)
        return data

    
    def get(self, geneid, **kwargs):
        ridx = self.geneids.index(geneid)
        data_gene = self[ridx]
        data = []
        for cidx, data_gene_sample in enumerate(data_gene):
            sample = self.samples[cidx]
            if all([getattr(sample, k)==v for k, v in kwargs.items()]):
                data.append(data_gene_sample)
        return data


    def save(self, filepath):
        dat = (self, self.geneids, self.samples) 
        fh = open(filepath, 'w')
        cPickle.dump(dat, fh)
        fh.close()

        
    @staticmethod
    def load(filepath):
        fh = open(filepath)
        dat, geneids, samples = cPickle.load(fh)
        fh.close()
        return Data(dat=dat, geneids=geneids, samples=samples)

        
