import math
import numpy as np
from enum import IntEnum

from sdirl.rl.pybrain_extensions import ParametricLoggingEpisodicTask, ParametricLoggingEnvironment

import logging
logger = logging.getLogger(__name__)


"""A multiple choice model.

Definition of the MDP.
"""

class State():
    """ State of MDP observed by the agent

    Parameters
    ----------
    options : list of Option
    focus : Focus
    select : Select
    """
    def __init__(self, option_values, option_comparisons, select):
        self.option_values = option_values
        self.option_comparisons = option_comparisons
        self.select = select

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (tuple([int(v) for v in self.option_values])
                + tuple([int(c) for c in self.option_comparisons])
                + (self.select, )).__hash__()

    def __repr__(self):
        return "({},{},{})".format(self.option_values, self.option_comparisons, self.select)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return State(self.option_values[:], self.option_comparisons[:], self.select)

class OptionValue(IntEnum):
    NOT_OBSERVED = 0
    ACCEPTABLE = 1
    NOT_ACCEPTABLE = 2

class OptionComparison(IntEnum):
    NOT_OBSERVED = 0
    LESS_THAN = 1
    EQUAL_TO = 2
    MORE_THAN = 3

class Select(IntEnum):
    NOT_SELECTED = 0
    HAS_SELECTED_1 = 1
    HAS_SELECTED_2 = 2
    HAS_SELECTED_3 = 3

class Action(IntEnum):
    OBSERVE_1 = 0
    OBSERVE_2 = 1
    OBSERVE_3 = 2
    SELECT_1 = 3
    SELECT_2 = 4
    SELECT_3 = 5

class ChoiceTask(ParametricLoggingEpisodicTask):

    def __init__(self, env, step_penalty, max_number_of_actions_per_session):
        super(ChoiceTask, self).__init__(env)
        #self.step_penalty = step_penalty
        self.max_number_of_actions_per_session = max_number_of_actions_per_session

    def to_dict(self):
        return {
                "ABC_params": self.v,
                "max_number_of_actions_per_session": self.max_number_of_actions_per_session,
                }

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        if self.env.state.select == Select.HAS_SELECTED_1:
            return self.env.draw_option_value(0)
        if self.env.state.select == Select.HAS_SELECTED_2:
            return self.env.draw_option_value(1)
        if self.env.state.select == Select.HAS_SELECTED_3:
            return self.env.draw_option_value(2)
        if self.env.state != self.env.prev_state:
            return 0.0  # acquired new information
        return self.v["step_penalty"]

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.state.select != Select.NOT_SELECTED:
            return True
        return False


class Option():
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def draw_value(self, random_state):
        if random_state.rand() < self.p:
            return self.v
        return 0.0

    def expected_value(self):
        return self.p * self.v

def identity(v):
    return v

class ChoiceEnvironment(ParametricLoggingEnvironment):

    def __init__(self,
            n_options=3,
            p_alpha=1.0, # ok
            p_beta=1.0, # ok
            v_loc=19.60, # ok
            v_scale=8.08, # ok
            v_df=100, # ok
            alpha=1.5, # ok
            calc_sigma=0.35, # ok
            tau_p=0.011, # ok
            tau_v=1.1, # ok
            f_err=0.1, # ok
            u_fun=identity,
            n_training_sets=10000):
        self.v = None # set with setup
        self.random_state = None # set with setup
        self.log = None # set by RL_model
        self.task = None # set by Task

        self.n_options = n_options
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.v_loc = v_loc
        self.v_scale = v_scale
        self.v_df = v_df
        #self.alpha = alpha
        #self.calc_sigma = calc_sigma
        #self.tau_p = tau_p
        #self.tau_v = tau_v
        #self.f_err = f_err
        self.u_fun = u_fun
        #self.tau_r = 2.0  # temp
        self.n_training_sets = n_training_sets
        assert self.n_options == 3

        self.options = list()
        self.training_sets = list()
        self.n_test_sets = 60
        self.next_test_id = 0
        self.training = True
        self.log_session_variables = ["options", "pair_index", "decoy_target", "decoy_type"]
        self.log_step_variables = ["action"]

        # technical variables
        self.discreteStates = True
        self.outdim = 1
        self.indim = 1
        self.discreteActions = True
        self.numActions = 6

    def to_dict(self):
        return {
                "n_options": self.n_options,
                "p_alpha": self.p_alpha,
                "p_beta": self.p_beta,
                "v_loc": self.v_loc,
                "v_scale": self.v_scale,
                "v_df": self.v_df,
                "ABC_params": self.v,
                "n_training_sets": self.n_training_sets,
                }

    def draw_option_value(self, index):
        baseline_value = float(np.mean([c.expected_value() for c in self.options]))
        return self.options[index].expected_value() - baseline_value

    def _get_options(self):
        if self.training is False:
            tid = self.next_test_id % self.n_test_sets
            pair_index = int(tid / 6)
            tid = tid % 6
            target = int(tid / 3)
            typ = tid % 3
            decoy_target = ["A", "B"][target]
            decoy_type = ["R", "F", "RF"][typ]
            self.next_test_id += 1
            return self._get_wedell_options(pair_index, decoy_target, decoy_type), pair_index, decoy_target, decoy_type
        if len(self.training_sets) >= self.n_training_sets:
            idx = self.random_state.randint(self.n_training_sets)
            return self.training_sets[idx], "T", "T", "T"
        options = list()
        for i in range(self.n_options):
            p = self.random_state.beta(self.p_alpha, self.p_beta)
            v = self.random_state.standard_t(self.v_df) * self.v_scale + self.v_loc
            # omit correlation r for now
            options.append(Option(p, v))
        self.training_sets.append(options)
        return options, "T", "T", "T"

    def _get_wedell_options(self, pair_index, decoy_target, decoy_type):
        # Wendell (1991): Distinguishing Among Models of Contextually Induced Preference Reversals
        A_choices = [
                {"O": Option(0.40, 25), "R": Option(0.40, 20), "F": Option(0.35, 25), "RF": Option(0.35, 20)},
                {"O": Option(0.50, 20), "R": Option(0.50, 18), "F": Option(0.45, 20), "RF": Option(0.45, 18)},
                {"O": Option(0.67, 15), "R": Option(0.67, 13), "F": Option(0.62, 15), "RF": Option(0.62, 13)},
                {"O": Option(0.83, 12), "R": Option(0.83, 10), "F": Option(0.78, 12), "RF": Option(0.78, 10)},
                ]
        B_choices = [
                {"O": Option(0.30, 33), "R": Option(0.25, 33), "F": Option(0.30, 30), "RF": Option(0.25, 30)},
                {"O": Option(0.40, 25), "R": Option(0.35, 25), "F": Option(0.40, 20), "RF": Option(0.35, 20)},
                {"O": Option(0.50, 20), "R": Option(0.45, 20), "F": Option(0.50, 18), "RF": Option(0.45, 18)},
                {"O": Option(0.67, 15), "R": Option(0.62, 15), "F": Option(0.67, 13), "RF": Option(0.62, 13)},
                ]
        all_options = [(0,0), (1,0), (2,0), (3,0), \
                              (1,1), (2,1), (3,1), \
                                     (2,2), (3,2), \
                                            (3,3)]
        idx = all_options[pair_index]
        options = [A_choices[idx[0]]["O"], B_choices[idx[1]]["O"]]
        if decoy_target == "A":
            options.append(A_choices[idx[0]][decoy_type])
        elif decoy_target == "B":
            options.append(B_choices[idx[1]][decoy_type])
        else:
            assert False
        return options

    def reset(self):
        # state hidden from agent
        self.options, self.pair_index, self.decoy_target, self.decoy_type = self._get_options()

        # state observed by agent
        self.state = State([OptionValue.NOT_OBSERVED]*3,
                           [OptionComparison.NOT_OBSERVED]*9,
                           Select.NOT_SELECTED)
        self.prev_state = self.state.copy()

        # misc environment state variables
        self.action = None
        self.n_actions = 0
        self._start_log_for_new_session()
        #if not self.training:
        #    print(self.state)

    def performAction(self, action):
        self.action = Action(int(action[0]))
        self.prev_state = self.state.copy()
        self.state = self.do_transition(self.state, self.action)
        self.n_actions += 1
        self._log_transition()
        #if not self.training:
        #    print(self.action)
        #    print(self.state)

    def _calculate(self, state, index):
        if state.option_values[index] != OptionValue.NOT_OBSERVED:
            # re-observing does not change previous estimate
            return state
        p = self.options[index].p ** self.v["alpha"]
        v = self.options[index].v
        u = self.u_fun(v)
        m = p * u + self.random_state.normal(0, self.v["calc_sigma"])
        if m > 0:
            state.option_values[index] = OptionValue.ACCEPTABLE
        else:
            state.option_values[index] = OptionValue.NOT_ACCEPTABLE
        return state

    def _compare(self, state, index1, index2):
        if state.option_values[index1] == OptionValue.NOT_OBSERVED or \
           state.option_values[index2] == OptionValue.NOT_OBSERVED:
            # can not compare before observing first
            return state
        assert index1 != index2, index1
        i1 = min(index1, index2)
        i2 = max(index1, index2)
        for kind in ["r", "p", "v"]:
            if kind == "p":
                f1 = self.options[i1].p
                f2 = self.options[i2].p
                tau = self.v["tau_p"]
                if i1 == 0 and i2 == 1:
                    idx = 0
                if i1 == 0 and i2 == 2:
                    idx = 1
                if i1 == 1 and i2 == 2:
                    idx = 2
            if kind == "v":
                f1 = self.options[index1].v
                f2 = self.options[index2].v
                tau = self.v["tau_v"]
                if i1 == 0 and i2 == 1:
                    idx = 3
                if i1 == 0 and i2 == 2:
                    idx = 4
                if i1 == 1 and i2 == 2:
                    idx = 5
            if kind == "r":
                p1 = self.options[i1].p ** self.v["alpha"]
                v1 = self.options[i1].v
                u1 = self.u_fun(v1)
                f1 = p1 * u1 + self.random_state.normal(0, self.v["calc_sigma"])
                p2 = self.options[i2].p ** self.v["alpha"]
                v2 = self.options[i2].v
                u2 = self.u_fun(v2)
                f2 = p2 * u2 + self.random_state.normal(0, self.v["calc_sigma"])
                tau = self.v["tau_r"]
                if i1 == 0 and i2 == 1:
                    idx = 6
                if i1 == 0 and i2 == 2:
                    idx = 7
                if i1 == 1 and i2 == 2:
                    idx = 8
            if state.option_comparisons[idx] == OptionComparison.NOT_OBSERVED:
                if f1 < f2 - tau:
                    state.option_comparisons[idx] = OptionComparison.LESS_THAN
                elif f1 < f2 + tau:
                    state.option_comparisons[idx] = OptionComparison.EQUAL_TO
                else:
                    state.option_comparisons[idx] = OptionComparison.MORE_THAN
                if self.random_state.rand() < self.v["f_err"]:
                    state.option_comparisons[idx] = self.random_state.choice(
                            [OptionComparison.LESS_THAN,
                             OptionComparison.EQUAL_TO,
                             OptionComparison.MORE_THAN])
        return state

    def do_transition(self, state, action):
        """ Changes the state of the environment based on agent action.
            Also depends on the unobserved state of the environment.

        Parameters
        ----------
        state : State
        action : Action

        Returns
        -------
        tuple (State, int) with new state and action duration in ms
        """
        state = state.copy()

        if action == Action.OBSERVE_1:
            state = self._calculate(state, 0)
        elif action == Action.OBSERVE_2:
            state = self._calculate(state, 1)
        elif action == Action.OBSERVE_3:
            state = self._calculate(state, 2)
        elif action == Action.SELECT_1:
            if state.option_values[0] != OptionValue.NOT_OBSERVED:
                state.select = Select.HAS_SELECTED_1
        elif action == Action.SELECT_2:
            if state.option_values[1] != OptionValue.NOT_OBSERVED:
                state.select = Select.HAS_SELECTED_2
        elif action == Action.SELECT_3:
            if state.option_values[2] != OptionValue.NOT_OBSERVED:
                state.select = Select.HAS_SELECTED_3
        else:
            raise ValueError("Unknown action: {}".format(action))

        if state.option_values[0] != OptionValue.NOT_OBSERVED and \
           state.option_values[1] != OptionValue.NOT_OBSERVED:
            state = self._compare(state, 0, 1)
        if state.option_values[0] != OptionValue.NOT_OBSERVED and \
           state.option_values[2] != OptionValue.NOT_OBSERVED:
            state = self._compare(state, 0, 2)
        if state.option_values[1] != OptionValue.NOT_OBSERVED and \
           state.option_values[2] != OptionValue.NOT_OBSERVED:
            state = self._compare(state, 1, 2)

        return state

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return [self.state.__hash__()]  # needs to return a list

