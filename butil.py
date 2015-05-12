"""
sequence: list, tuple, numpy.ndarray, set (?)
mapping: dict, OD, KeyedList
"""

import itertools
import cPickle 

import numpy as np
import pandas as pd


def all_same(seq):
    return all(elem == seq[0] for elem in seq)


def get_subseq(seq, idxs=None):
    if idxs is None:
        return seq
    else:
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
    if idxs is not None:
        keys = [mapping.keys()[idx] for idx in idxs]
    if f_key:
        keys = [k for k in mapping.keys() if f_key(k)]
    if f_value:
        keys = [k for k, v in mapping.items() if f_value(v)]
    
    vals = get_values(mapping, keys, default=default)
    return type(mapping)(zip(keys, vals))
    
    
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


def flatten(seq_nested, D=float('inf')):
    """
    Flatten a nested seq by any given depth.
    D: depth of flattening (flattened to the deepest possible by default);
       1 for one level down.
    """
    l_nested = list(seq_nested)
    d = 1  # current depth
    while d <= D:
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


def set_global(d):
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
    
    

    
