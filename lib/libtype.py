"""
seq: sequence type, e.g., list, tuple, set (?)
mapping: mapping type, e.g., dict, OD, KeyedList
"""

import itertools
import numpy as np

def all_same(seq):
    return all(elem == seq[0] for elem in seq)


def get_subseq(seq, idxs=None):
    if idxs is None:
        return seq
    else:
        return type(seq)([seq[idx] for idx in idxs])


def get_values(mapping, keys=None, default=None):
    if keys is None:
        return mapping.values()
    else:
        return type(keys)([mapping.get(key, default) for key in keys])


def get_submapping(mapping, keys=None):
    if keys is None:
        return mapping
    else:
        vals = get_values(mapping, keys)
        return type(mapping)(zip(keys, vals))
    
    
def change_keys(mapping, func):
    keys, values = mapping.keys(), mapping.values() 
    keys_new = [func(key) for key in keys]
    return type(mapping)(zip(keys_new, values))


def change_values(mapping, func):
    keys, values = mapping.keys(), mapping.values() 
    values_new = [func(value) for value in values]
    return type(mapping)(zip(keys, values_new))


def change_items(mapping, func_key=None, func_value=None):
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
    return type(mapping1)(mapping1.items()+mapping2.items()) 


def twomappings2one(mapping1, mapping2, func):
    """
    """
    if mapping1.keys() != mapping2.keys():
        raise StandardError("The two input mappings have different keys.")
    vals = [func(mapping1.get(key), mapping2.get(key)) 
            for key in mapping1.keys()]
    return type(mapping1)(zip(mapping1.keys(), vals))


def flatten_shallow(seq_nested):
    """
    Only flatten a shallow nested sequence, such as [[1,2], [3]]. 
    """
    chain = itertools.chain(*seq_nested)
    return type(seq_nested)(chain)


def get_closest_index(seq, elem):
    """
    Return the index of the element in the sequence that is closest
    to the given element; if multiple elements are closest, then
    only the first occurring element is returned. 
    """
    diffs = np.abs(np.array(seq) - elem)
    return np.where(diffs == np.min(diffs))[0][0]


def format(d, digit=1, e=False):
    """
    Input:
        d: a digit, either a float or an int
        e: a flag for whether to use scientific notation
    """
    if e:
        s = ('%.'+str(digit)+'e')%d
        # remove the extra "0" in the scientific notation
        return s.replace('e+0', 'e+').replace('e-0', 'e-')
    else:
        if isinstance(d, int):
            return d
        else:
            if int(d) - d == 0:
                return int(d)
            else:
                return float(('%.'+str(digit)+'g')%d)


def index(seq, elem, cutoff=1e-6):
    """
    """
    def true_if_close(obj1, obj2, cutoff=cutoff):
        try:
            if np.max(np.abs(np.array(obj1) - np.array(obj2))) < cutoff:
                return True
            else:
                return False
        except:
            if obj1 == obj2:
                return True
            else:
                return False
    bools = [true_if_close(e, elem) for e in seq]
    return bools.index(True)