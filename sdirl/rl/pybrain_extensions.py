import numpy as np
import scipy as sp

from pybrain.structure.modules.module import Module
from pybrain.rl.explorers.discrete import EpsilonGreedyExplorer
from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Extensions to the pybrain library.
"""

class SparseActionValueTable(ActionValueTable):
    """ Sparse version of the ActionValueTable from pybrain, uses less memory.
    """

    def __init__(self, numActions, random_state, name=None):
        Module.__init__(self, 1, 1, name)
        self.n_actions = numActions
        self.numColumns = numActions
        self.random_state = random_state
        if isinstance(self, Module) or isinstance(self, Connection):
            self.hasDerivatives = True
        if self.hasDerivatives:
            self._derivs = None
        self.randomize()
        self._params = None

    def randomize(self):
        self.sparse_params = dict() # dictionary-of-rows sparse matrix
        self.initval = None

    def initialize(self, value=1e-5):
        self.initval = value

    def getMaxAction(self, state):
        values = self.getActionValues(state)
        action = sp.where(values == max(values))[0]
        return self.random_state.choice(action)

    def check_bounds(self, row=None, column=None):
        if row is not None and (row < 0):
            raise ValueError("Row out of bounds (was {})".format(row))
        if column is not None and (column < 0 or column >= self.n_actions):
            raise ValueError("Column out of bounds (was {})".format(column))

    def getValue(self, row, column):
        return self.getActionValues(row)[column]

    def _init_or_random_val(self):
        if self.initval is None:
            # From ParameterContainer.randomize()
            return self.random_state.randn() * self.stdParams
        else:
            # From ActionValueTable.initialize()
            return self.initval

    def getActionValues(self, state):
        self.check_bounds(state)
        if state is None:
            return None
        r = self.sparse_params.get(state, None)
        if r is None:
            r = np.array([float(self._init_or_random_val()) for i in range(self.n_actions)])
            self.sparse_params[state] = r
        return r[:]

    def updateValue(self, row, column, value):
        self.check_bounds(row, column)
        if row is None or column is None:
            return
        r = self.getActionValues(row)
        r[column] = value
        self.sparse_params[row] = r

    def mutate(self):
        raise NotImplementedError("This should not be called.")

    def derivs(self):
        raise NotImplementedError("This should not be called.")

    def resetDerivatives(self):
        raise NotImplementedError("This should not be called.")



class EpisodeQ(Q):
    """ A slight modification of the pybrain Q learner to add special handling
        for the end state of the session.
    """

    counter = 0

    def learn(self):
        sdq = list()
        if self.batchMode:
            samples = self.dataset
        else:
            samples = [[self.dataset.getSample()]]

        for seq in samples:
            # information from the previous episode (sequence)
            # should not influence the training on this episode
            self.laststate = None
            self.lastaction = None
            self.lastreward = None

            for state, action, reward in seq:

                state = int(state)
                action = int(action)

                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    self.lastreward = reward
                    continue

                qvalue = self.module.getValue(self.laststate, self.lastaction)
                maxnext = self.module.getValue(state, self.module.getMaxAction(state))
                dq = self.alpha * (self.lastreward + self.gamma * maxnext - qvalue)
                sdq.append(abs(dq))
                self.module.updateValue(self.laststate, self.lastaction, qvalue + dq)

                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward

            # Add the missing update step for the final action
            qvalue = self.module.getValue(self.laststate, self.lastaction)
            dq = self.alpha * (self.lastreward - qvalue)
            sdq.append(abs(dq))
            self.module.updateValue(self.laststate, self.lastaction, qvalue + dq)



class EGreedyExplorer(DiscreteExplorer):
    """ Reimplementation of rllib EpsilonGreedyExplorer to use the provided random_state
    """
    def __init__(self, random_state, epsilon = 0.3, decay = 0.9999):
        DiscreteExplorer.__init__(self)
        self.random_state = random_state
        self.epsilon = epsilon
        self.decay = decay
        self.module = None

    def _forwardImplementation(self, inbuf, outbuf):
        assert self.module is not None

        if self.random_state.rand() < self.epsilon:
            outbuf[:] = np.array([self.random_state.randint(self.module.numActions)])
        else:
            outbuf[:] = inbuf

        self.epsilon *= self.decay

