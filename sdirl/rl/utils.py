import numpy as np

import logging
logger = logging.getLogger(__name__)

# pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

""" RL related utility functions and classes
"""

class InitialStateGenerator():

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.n_initial_states = 0  # number of possible initial states

    def get_random_initial_state_id(self, random_state):
        """ Returns a random intial state id

        Parameters
        ----------
        random_state : np.random.RandomState
        """
        if self.n_initial_states <= 0:
            raise ValueError("Must have at least one possible initial state.")
        return random_state.randint(self.n_initial_states)

    def get_initial_state(self, id_number):
        """ Returns an initial state corresponding to id_number

        Parameters
        ----------
        id_number : int
            in [0, n_initial_states)
        """
        raise NotImplementedError("Subclass implements")

    def to_dict(self):
        return {
                "class": self.__class__.__name__,
                "grid_size": self.grid_size,
                "n_initial_states": self.n_initial_states,
                }


class Path():
    def __init__(self, transitions):
        self.transitions = transitions

    def append(self, transition):
        self.transitions.append(transition)

    def get_start_state(self):
        if len(self) < 1:
            raise ValueError("Path contains no transitions and thus no start state")
        return self.transitions[0].prev_state

    def __eq__(a, b):
        if len(a) != len(b):
            return False
        for t1, t2 in zip(a.transitions, b.transitions):
            if t1 != t2:
                return False
        return True

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        ret = list()
        for t in self.transitions:
            ret.append("{};".format(t))
        return "".join(ret)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Path([transition.copy() for transition in self.transitions])


class Transition():
    def __init__(self, prev_state, action, next_state):
        self.prev_state = prev_state  # assume object
        self.action = action  # assume enum
        self.next_state = next_state  # assume object

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.prev_state, self.action, self.next_state).__hash__()

    def __repr__(self):
        return "T({}+{}->{})".format(self.prev_state, self.action, self.next_state)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Transition(self.prev_state.copy(), self.action, self.next_state.copy())


class PathTreeIterator():
    """ Iterator for a path tree

    Parameters
    ----------
    root : Observation
    paths : dict[observation] = node with nodes being list of tuples (transition, next observation)
    """
    def __init__(self, root, paths, maxlen):
        self.root = root
        self.paths = paths
        self.maxlen = maxlen

    def __iter__(self):
        self.indices = [0] * self.maxlen
        self.end = False
        return self

    def __next__(self):
        if self.end is True:
            raise StopIteration()
        path = Path([])
        node = self.paths[self.root]
        nvals = list()
        for i in self.indices:
            nvals.append(len(node))
            transition, next_obs = node[i]
            path.append(transition)
            assert next_obs in self.paths, "Observation {} not found in tree?".format(next_obs)
            node = self.paths[next_obs]
        for i in reversed(range(len(self.indices))):
            if self.indices[i] < nvals[i]-1:
                self.indices[i] += 1
                break
            else:
                self.indices[i] = 0
        if max(self.indices) == 0:
            self.end = True
        return path


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

