import numpy as np
import scipy as sp

from sdirl.rl.utils import Path, Transition

from pybrain.structure.modules.module import Module
from pybrain.rl.environments import Environment, EpisodicTask
from pybrain.rl.explorers.discrete import EpsilonGreedyExplorer
from pybrain.rl.explorers.discrete.discrete import DiscreteExplorer
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Extensions to the pybrain library.
"""

class ParametricLoggingEpisodicTask(EpisodicTask):
    """ Extension of the basic episodic task with tunable parameters
        and better support for logging.
    """

    def __init__(self, env):
        super(ParametricLoggingEpisodicTask, self).__init__(env)
        self.env.task = self
        self.v = None

    def setup(self, variables):
        """ Set the variables of the task
        """
        self.v = variables


class ParametricLoggingEnvironment(Environment):
    """ Extension of the basic environment with tunable parameters
        and better support for logging.
    """

    def __init__(self):
        super(ParametricLoggingEnvironment, self).__init__()
        self.v = None
        self.log = None
        self.task = None  # set by task
        self.state = None
        self.prev_state = None
        self.action = None
        self.log_session_variables = list()  # logged at start of session
        self.log_step_variables = list()  # logged after each step

    def setup(self, variables, random_state):
        """ Finishes the initialization
        """
        self.v = variables
        self.random_state = random_state
        self.reset()

    def start_logging(self):
        self.log = dict()

    def end_logging(self):
        self.log = None

    def _start_log_for_new_session(self):
        """ Set up log when new session starts
        """
        if self.log != None:
            if "session" not in self.log:
                self.log["session"] = 0
                self.log["sessions"] = [dict()]
            else:
                self.log["session"] += 1
                self.log["sessions"].append(dict())
            self.step_data = self.log["sessions"][self.log["session"]]
            for varname in self.log_session_variables:
                self.step_data[varname] = getattr(self, varname)
            self.step_data["path"] = Path([])
            self.step_data["rewards"] = list()
            for varname in self.log_step_variables:
                self.step_data[varname] = list()

    def _log_transition(self):
        """ Should be called after transition
        """
        if self.log != None:
            self.step_data["path"].append(Transition(self.prev_state, self.action, self.state))
            self.step_data["rewards"].append(self.task.getReward())
            for varname in self.log_step_variables:
                self.step_data[varname].append(getattr(self, varname))


class SparseActionValueTable(ActionValueTable):
    """ Sparse version of the ActionValueTable from pybrain, uses less memory.
    """

    def __init__(self, numActions, random_state, softq=False, name=None):
        Module.__init__(self, 1, 1, name)
        self.n_actions = numActions
        self.numColumns = numActions
        self.random_state = random_state
        self.softq = softq
        if isinstance(self, Module) or isinstance(self, Connection):
            self.hasDerivatives = True
        if self.hasDerivatives:
            self._derivs = None
        self.randomize()
        self._params = None

    def _forwardImplementation(self, inbuf, outbuf):
        """ Take a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        if self.softq is False:
            outbuf[0] = self.getMaxAction(inbuf[0])
        else:
            outbuf[0] = self.getSoftMaxAction(inbuf[0])

    def randomize(self):
        self.sparse_params = dict() # dictionary-of-rows sparse matrix
        self.initval = None

    def initialize(self, value=1e-5):
        self.initval = value

    def getMaxAction(self, state):
        values = self.getActionValues(state)
        action = sp.where(values == max(values))[0]
        return self.random_state.choice(action)

    def getSoftMaxAction(self, state):
        values = self.getActionValues(state)
        exp_val = [np.exp(v) for v in values]
        norm_exp_val = exp_val / sum(exp_val)
        cum_norm_exp_val = np.cumsum(norm_exp_val)
        rnd = self.random_state.rand()
        for idx, lim in enumerate(cum_norm_exp_val):
            if rnd < lim:
                return idx
        assert False

    def check_bounds(self, column=None):
        if column < 0 or column >= self.n_actions:
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
        if state is None:
            return None
        r = self.sparse_params.get(state, None)
        if r is None:
            r = np.array([float(self._init_or_random_val()) for i in range(self.n_actions)])
            self.sparse_params[state] = r
        return r[:]

    def updateValue(self, row, column, value):
        self.check_bounds(column)
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

