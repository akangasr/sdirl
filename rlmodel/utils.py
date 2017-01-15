import numpy as np

import logging
logger = logging.getLogger(__name__)

# pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Misc utility functions, mostly related to transformations between state representations.
"""

def is_integer(i):
    """ Returns true if the type of 'i' is some kind of common integer.
    """
    return isinstance(i, (int, np.intc, np.intp, np.int8, np.int16,
                          np.int32, np.int64, np.uint8, np.uint16,
                          np.uint32, np.uint64))


def n_states_of_vec(l, nval):
    """ Returns the amount of different states a vector of length 'l' can be
        in, given that each index can be in 'nval' different configurations.
    """
    if type(l) != int or type(nval) != int or l < 1 or nval < 1:
        raise ValueError("Both arguments must be positive integers.")
    return nval ** l


def vec_state_to_scalar(vec, nval):
    """ Converts a vector state into a enumerated scalar state.
        The function 'scalar_state_to_vec' is the inverse of this.
    """
    if not is_integer(nval):
        raise ValueError("Type of nval must be int.")
    s = 0
    i = 0
    for v in vec:
        if not is_integer(v) or v < 0 or v >= nval:
            raise ValueError("Vector values must be integers in range"
                             "[0, {}] (was: {} ({}))."
                             .format(nval-1, v, type(v)))
        s += (nval ** i) * v
        i += 1
    return int(s)


def scalar_state_to_vec(state, l, nval):
    """ Converts a scalar state into a vector of length 'l' with 'nval'
        different values for each index.
    """
    if not is_integer(l) or not is_integer(nval) or l < 1 or nval < 1:
        raise ValueError("Both 'l' and 'nval' must be positive integers"
                         "(was {} ({}), {} ({}))."
                         .format(l, type(l), nval, type(nval)))
    if not is_integer(state) or state < 0 or state >= n_states_of_vec(l, nval):
        raise ValueError("State must be integer and within reasonable"
                         "bounds (was {} ({}))."
                         .format(state, type(state)))
    i = 0
    vec = np.zeros(l, dtype=np.int64)
    while state > 0:
        val = state % nval
        vec[i] = int(val)
        state -= val
        state /= nval
        i += 1
    return vec.tolist()

