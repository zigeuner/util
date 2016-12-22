"""
sequence: list, tuple, numpy.ndarray, set (?)
mapping: dict, OD, KeyedList
"""

import itertools
import cPickle 
import os
import copy
from collections import OrderedDict as OD, Mapping

import numpy as np
import pandas as pd


def all_same(seq):
    return all(elem == seq[0] for elem in seq)


def get_subseq(seq, idxs):
    return type(seq)([seq[idx] for idx in idxs])


def get_values(mapping, keys, default=None):
    return type(keys)([mapping.get(key, default) for key in keys])


def get_submapping(mapping, keys=None, idxs=None, f_key=None, f_value=None,
                   default=None):
    """
    
    Input:
        keys:
        idxs: indices, works for ordered mappings such as OrderedDict
        f_key: a function that takes a key and returns True/False
        f_value: a function that takes a value and returns True/False
    """
    if keys is not None:
        ks = [k for k in keys if k in mapping]
    if idxs is not None:
        ks = [mapping.keys()[idx] for idx in idxs]
    if f_key:
        ks = [k for k in mapping.keys() if f_key(k)]
    if f_value:
        ks = [k for k, v in mapping.items() if f_value(v)]
    vs = get_values(mapping, ks)
    return type(mapping)(zip(ks, vs))
    
    
def chkeys(mapping, func):
    """
    """
    keys, vals = mapping.keys(), mapping.values() 
    keys2 = [func(key) for key in keys]
    return type(mapping)(zip(keys2, vals))


def chvals(mapping, func):
    """
    """
    keys, vals = mapping.keys(), mapping.values() 
    vals2 = [func(v) for v in vals]
    return type(mapping)(zip(keys, vals2))


def chitems(mapping, func_key=None, func_value=None):
    """
    More general than change_keys and change_values. 
    """
    keys, values = mapping.keys(), mapping.values()
    if func_key: 
        keys_new = [func_key(key) for key in keys]
    else:
        keys_new = keys
    if func_value:
        values_new = [func_value(value) for value in values]
    else:
        values_new = values
    return type(mapping)(zip(keys_new, values_new))


def merge_mappings(mapping1, mapping2):
    return type(mapping1)(mapping1.items() + mapping2.items()) 


def twomappings2one(mapping1, mapping2, func):
    """
    """
    if mapping1.keys() != mapping2.keys():
        raise StandardError("The two input mappings have different keys.")
    vals = [func(mapping1.get(key), mapping2.get(key)) 
            for key in mapping1.keys()]
    return type(mapping1)(zip(mapping1.keys(), vals))


def flatten(seq_nested, depth=float('inf')):
    """
    Change D to depth (FIXME *)
    
    Flatten a nested seq by any given depth.
    depth: depth of flattening (flattened to the deepest possible by default);
       1 for one level down.
    """
    l_nested = list(seq_nested)
    d = 1  # current depth
    while d <= depth:
        out = []
        for elem in l_nested:
            if hasattr(elem, '__iter__'):
                out.extend(elem)
            else:
                out.append(elem)
        if l_nested == out:
            break
        l_nested = out
        d += 1
    out = type(l_nested)(out)
    return out


def rmap(func_or_dict, seq):
    """
    A recursive version of Python builtin 'map' that also works for 
    nested 'seq'.
    """
    if hasattr(func_or_dict, '__getitem__'):  # dict and OD
        func = lambda k: func_or_dict.get(k, k)
    else:
        func = func_or_dict
    seq2 = []
    for elem in seq:
        if hasattr(elem, '__iter__'):
            seq2.append(rmap(func, elem))
        else:
            seq2.append(func(elem))
    seq2 = type(seq)(seq2)
    return seq2
        

def format(num, ndigit=1, e=False):
    """
    Format a number to return its:
    1) scientific notation (str) with ndigit if e is True;
    2) float with ndigit (or less if trailing zeros) significant digit;
    3) integer if trailing all zeros after the decimal point.
    
    Input:
        num: a number, either a float or an int
        e: a flag for whether to use scientific notation
    """
    if e:
        s = ('%.' + str(ndigit) + 'e') % num
        # remove the extra "0" in the scientific notation
        return s.replace('e+0', 'e+').replace('e-0', 'e-')
    else:
        if isinstance(num, int):
            return num
        else:
            num = float(('%.'+str(ndigit)+'f') % num)
            if int(num) == num:
                return int(num)
            else:
                return num


def true_if_close(obj1, obj2, cutoff=1e-6):
    """
    This function returns True if either two numbers, 
    or the maximally different numbers of two sequences of the same length, 
    are close enough (as determined by the cutoff); otherwise returns False.
    """
    try:
        if np.max(np.abs(np.array(obj1) - np.array(obj2))) < cutoff:
            return True
        else:
            return False
    except ValueError:  # obj1 and obj2 are of different lengths
        return False
                
            
def get_indices_one(seq, num, cutoff=None, k=None):
    """
    This function returns the indices of one of the following three cases:
    a) elements in seq that are equal to num (k=None, cutoff=None)
    b) elements in seq that are close to num within the cutoff
    c) k elements in seq that are closest to num
    """ 
    seq = np.array(seq)
    idxs = np.where(seq == num)[0]
    if num == np.inf:
        return idxs 
    if len(idxs)==0 or k or cutoff:
        if k:
            diffs = zip(range(len(seq)), np.abs(seq - num))
            diffs_sort = np.array(sorted(diffs, key=lambda x:x[1]))
            idxs = [diff[0] for diff in diffs_sort[:k]]
        if cutoff:
            idxs = np.where(np.abs(seq - num) < cutoff)[0].tolist()
    return idxs


def get_indices(seq, nums, cutoff=None, k=None):
    if hasattr(nums, '__iter__'):
        idxs = []
        for num in nums:
            idxs.extend(get_indices_one(seq, num, cutoff=cutoff, k=k))
    else:
        idxs = get_indices_one(seq, nums, cutoff=cutoff, k=k)
    return idxs


def powerset(iterable):
    """
    This function returns a list of all subsets (tuples) of an iterable.
    """
    pset = []
    for size in range(len(iterable) + 1):
        pset.extend(list(itertools.combinations(iterable, size)))
    return pset


def get_product(*iterables):
    return pd.MultiIndex.from_product(iterables).tolist()


def to_pickle(obj, filepath):
    """
    """
    fh = open(filepath, 'wb')
    cPickle.dump(obj, fh)
    fh.close()


def read_pickle(filepath):
    """
    """
    fh = open(filepath, 'rb')
    obj = cPickle.load(fh)
    fh.close()
    return obj


def set_global(**d):
    import __builtin__
    for obid, ob in d.items():
        __builtin__.__dict__[obid] = ob
        

def pprint(ob):
    if hasattr(ob, 'items'):
        for key, value in ob.items():
            print key, value
    else:
        for elem in ob:
            print elem


import pprocess as pp
def map_parallel(func, inputs):
    nproc = pp.get_number_of_cores()
    outputs = pp.pmap(func, inputs, limit=nproc)
    if type(inputs) == np.ndarray:
        outputs = np.array(list(outputs))
    else:
        outputs = type(inputs)(outputs)
    return outputs
    


def check_filepath(filepath):
    """
    The function performs two tasks:
        1. Check if the directory exists, and create it if not
        2. Check if the filepath exists, and ask for permission if yes
    """
    # task 1
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # task 2
    if os.path.isfile(filepath):
        return raw_input("%s exists: Proceed? "%filepath) in\
            ['y', 'Y', 'yes', 'Yes', 'YES']
    else:
        return True
    
    
def venn(seq1, seq2):
    """
    """
    _order = lambda s, seq: type(seq)([_ for _ in seq if _ in s])
    
    s1, s2 = set(seq1), set(seq2)
    intersection = _order(set.intersection(s1, s2), seq1)
    s1_specific = _order(s1 - s2, seq1)
    s2_specific = _order(s2 - s1, seq2)
    return intersection, s1_specific, s2_specific


def merge_dicts(*dicts):
    """Merge the input dictionaries and return the merged.
    # http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    """
    merger = dict()
    for d in dicts:
        merger.update(d) 
    return merger


class Series(pd.Series):
    """
    Reference:
    http://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures
    """
    @property
    def _constructor(self):  
        return Series


    @property
    def _constructor_expanddim(self):
        return DF
        
    
    def __init__(self, data=None, index=None, **kwargs):
        """
        Overwrite the constructor...
        
        >>> ser = pd.Series([1,2], index=['A','B'])
        >>> ser2 = pd.Series(ser, index=['a','b'])
        >>> ser2
            a   NaN
            b   NaN
            dtype: float64
        >>> ser3 = Series(ser, index=['a','b'])
        >>> ser3
            a   1
            b   2
        """
        if data is None:
            data  = []
        if index is None and hasattr(data, 'index') and\
                not callable(data.index):
            index = data.index
        if hasattr(data, 'values') and not callable(data.values):
            data = data.values
        # FIXME ***
        super(Series, self).__init__(data, index=index, **kwargs)
        
    
    def copy(self, **kwargs):
        """
        Copy custom attributes as well.
        """
        ser_cp = super(Series, self).copy(**kwargs)
        ser_cp.__dict__ = self.__dict__
        return ser_cp
    
    
    def keys(self):
        return self.index.tolist()
    
    @property
    def varids(self):
        return self.index.tolist()
        
    
    @property
    def logvarids(self):
        return map(lambda varid: 'log_'+varid, self.varids)
    
        
    def log(self):
        return Series(np.log(self), self.logvarids)
    
    def log10(self):
        raise NotImplementedError("")
    
    
    def exp(self):
        if not all([varid.startswith('log_') for varid in self.varids]):
            raise ValueError("The values are not in log-scale.")
        return Series(np.exp(self), 
                      map(lambda varid: varid.lstrip('log_'), self.varids))


    def randomize(self, seed=None, distribution='lognormal', **kwargs):
        """
        Input:
            distribution: 'lognormal', 'normal', etc.
                If 'lognormal': default is sigma=1
            kwargs: sigma                
        """
        if seed is not None:
            np.random.seed(seed)
        if distribution == 'lognormal':
            if 'sigma' not in kwargs:
                kwargs['sigma'] = 1
            x = self * np.random.lognormal(size=self.size, **kwargs)
        if distribution == 'normal':
            x = self + np.random.normal(size=self.size, **kwargs)
        return x 
    perturb = randomize  # deprecation warning FIXME *
    
    
    # self[varids] 
    #def reorder(self, varids):
    #    """
    #    """
    #    if set(self.varids) != set(varids):
    #        raise ValueError("The parameter ids are different:\n%s\n%s"%\
    #                         (str(self.varids), str(varids)))
    #    return Series(self, varids)
    
        
    def to_od(self):
        return OD(self)
    
    def append(self, *args, **kwargs):
        return Series(super(Series, self).append(*args, **kwargs))

    # FIXME *: why returning pd.Series instead?
    #def append(self, other, **kwargs):
    #    return Series(super(Series, self).append(other, **kwargs))
    
    def items(self):
        return list(self.iteritems())
    
    
    def filt(self, f_key=None, f_val=None):  # FIXME ***: ugly name...
        return Series(get_submapping(self.to_od(), f_key=f_key, f_value=f_val))
    
    
    def multiindexify(self, index_new, add_to='left', names=None, in_place=True):
        """Make the index a multiindex by adding a new dimension (to the 
        left or right). The new dimension of index is specified by index_new, which
        is either str/number or a dictionary mapping from the old index to the
        new dimension. 
        
        index_new: tuplized
        
        """
        if isinstance(index_new, Mapping):
            if add_to == 'left':
                #index_new[tu_] = ('a','b')
                #index_new[tu_] = (('a','b'),)
                
                tus = [index_new[idx_]+(idx_,) for idx_ in self.index]
            else:
                tus = [(idx_,)+index_new[idx_] for idx_ in self.index]
        else:
            if add_to == 'left':
                tus = [index_new+(idx_,) for idx_ in self.index]
            else:
                tus = [(idx_,)+index_new for idx_ in self.index]
        if in_place:
            self.index = pd.MultiIndex.from_tuples(tus, names=names)
        else:
            return Series(self, pd.MultiIndex.from_tuples(tus, names=names))
            
    
class DF(pd.DataFrame):
    """
    Reference:
    
    http://stackoverflow.com/questions/13460889/how-to-redirect-all-methods-of-a-contained-class-in-python
    
    http://stackoverflow.com/questions/22155951/how-to-subclass-pandas-dataframe
    http://stackoverflow.com/questions/29569005/error-in-copying-a-composite-object-consisting-mostly-of-pandas-dataframe
    
    http://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures
    """
    
    @property
    def _constructor(self, **kwargs):
        return self.__class__
    
    
    @property
    def _constructor_sliced(self):
        return Series
    
    """
    def __init__(self, data, index=None, columns=None, **kwargs):
        if index is None and hasattr(data, 'index') and\
                not callable(data.index):
            index = data.index
        if columns is None and hasattr(data, 'columns'):
            columns = data.columns
        if hasattr(data, 'values') and not callable(data.values):
            data = data.values
        # FIXME ***
        #print id(self)
        #print id(self.__class__)
        #print id(DF)
        super(DF, self).__init__(data, index=index, columns=columns, **kwargs)
    """
    
    def get_rowvarids(self):
        return self.index.tolist()
    
    def set_rowvarids(self, rowvarids):
        self.index = pd.Index(rowvarids)
    
    rowvarids = property(get_rowvarids, set_rowvarids)
    
    def get_colvarids(self):
        return self.columns.tolist()
    
    def set_colvarids(self, colvarids):
        self.columns = pd.Index(colvarids)
    
    colvarids = property(get_colvarids, set_colvarids)
    
    @property
    def nrow(self):
        return self.shape[0]
        
    @property
    def ncol(self):
        return self.shape[1]
    
    
    def copy(self):
        """
        Copy custom attributes as well.
        """
        df_cp = copy.deepcopy(self)
        df_cp.__dict__ = self.__dict__
        return df_cp
    
    
    #def append(self, ser, varid, axis=0):
    #    """
    #    """
    #    pass
    
    
    #def extend(self, other, axis=0):
    #    """
    #    """
    #    pass
    
    
    def dump(self, filepath):
        """
        Dump custom attributes as well.
        """
        fh = open(filepath, 'w')
        cPickle.dump((self, self.__dict__), fh)
        fh.close()
    
    
    def load(self, filepath):
        # dat, attrs = cPickle.load(fh)
        # mydf = self.__class__(dat)  -- does it work??
        # mydf.__dict__ = attrs 
        pass
    
    
    # self.to_csv
    #def save(self):
    #    """
    #    Just data (no metadata)
    #    Human readable
    #    """
    #    pass
    
    
    # self.iloc[::-1]
    # self.iloc[:, ::-1]
    #def flip(self, axis=0):
    #    """
    #    Input:
    #        axis: 0 (up-down) or 1 (left-right)
    #    """
    #    if axis == 0:
    #        return DF(np.flipud(self), self.rowvarids[::-1], self.colvarids)
    #    if axis == 1:
    #        return DF(np.fliplr(self), self.rowvarids, self.colvarids[::-1])
    
    
    def to_series(self):
        """
        """
        dat = self.values.flatten()
        index = list(itertools.product(self.rowvarids, self.colvarids))
        return Series(dat, index=index)
    
    
    #def pca(self):
    #    pass
    
    
    def plot(self, fmt='heatmap', orientation=''):
        """
        heatmap or table
        """
        pass







