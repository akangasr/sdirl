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

class Action(IntEnum):
    SELECT_1 = 0
    SELECT_2 = 1
    SELECT_3 = 2

class ChoiceTask(ParametricLoggingEpisodicTask):

    def __init__(self, env, max_number_of_actions_per_session):
        super(ChoiceTask, self).__init__(env)
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
        if self.env.state.select >= 0:
            return self.env.draw_option_value(self.env.state.select)
        return self.v["step_penalty"]

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.state.select >= 0:
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


class ChoiceEnvironment(ParametricLoggingEnvironment):

    def __init__(self,
            n_options=3,
            p_alpha=1.0, # ok
            p_beta=1.0, # ok
            v_loc=19.60, # ok
            v_scale=8.08, # ok
            v_df=100, # ok
            reward_type="utility",
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
        self.numActions = 3

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
        return self.options[index].expected_value()

    def utility_function(self, option):
        return np.power(option.p, self.v["alpha"]) * option.v

    def estimate_value(self, option, precise=False):
        if not precise:
            return float(self.random_state.normal(self.utility_function(option), self.v["calc_sigma"]))
        return self.utility_function(option)

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
        umin = min(self.utility_function(Option(0.78, 10)),
                   self.utility_function(Option(0.25, 30))) - self.v["calc_sigma"]*3
        umax = min(self.utility_function(Option(0.83, 12)),
                   self.utility_function(Option(0.30, 33))) + self.v["calc_sigma"]*3
        for i in range(self.n_options):
            while True:
                p = self.random_state.beta(self.p_alpha, self.p_beta)
                v = max(0.0, self.random_state.standard_t(self.v_df) * self.v_scale + self.v_loc)
                opt = Option(p, v)
                u = self.utility_function(opt)
                if u > umin and u < umax:
                    # this sample would probably not affect end results measured with wedell set
                    # as the absolute value of the estimated utility is a state variable
                    break
            # omit correlation r for now
            #p = self.random_state.uniform(0.1, 0.9)
            #v = self.random_state.uniform(5, 40)
            options.append(opt)
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
        self.state = State([-9999]*3, [-9999]*6, -1)
        self.prev_state = self.state.copy()

        # misc environment state variables
        self.action = None
        self.n_actions = 0
        self._start_log_for_new_session()
        #if not self.training:
        #    print(self.state)

        u1 = self.estimate_value(self.options[0])
        u2 = self.estimate_value(self.options[1])
        u3 = self.estimate_value(self.options[2])

        self.state.option_values[0] = int(u1 * self.v["uti_scale"])
        self.state.option_values[1] = int(u2 * self.v["uti_scale"])
        self.state.option_values[2] = int(u3 * self.v["uti_scale"])
        self.state.option_comparisons[0] = self._compare(self.options[0].p, self.options[1].p, self.v["tau_p"])
        self.state.option_comparisons[1] = self._compare(self.options[0].p, self.options[2].p, self.v["tau_p"])
        self.state.option_comparisons[2] = self._compare(self.options[1].p, self.options[2].p, self.v["tau_p"])
        self.state.option_comparisons[3] = self._compare(self.options[0].v, self.options[1].v, self.v["tau_v"])
        self.state.option_comparisons[4] = self._compare(self.options[0].v, self.options[2].v, self.v["tau_v"])
        self.state.option_comparisons[5] = self._compare(self.options[1].v, self.options[2].v, self.v["tau_v"])


    def performAction(self, action):
        self.action = Action(int(action[0]))
        self.prev_state = self.state.copy()
        self.state = self.do_transition(self.state, self.action)
        self.n_actions += 1
        self._log_transition()
        #if not self.training:
        #    print(self.action)
        #    print(self.state)

    def _compare(self, v1, v2, threshold):
        if self.random_state.uniform(0, 1) < self.v["comp_err"]:
            return self.random_state.randint(3)
        if v1 < v2 - threshold:
            return 0
        elif v1 < v2 + threshold:
            return 1
        else:
            return 2

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

        if action == Action.SELECT_1:
            state.select = 0
        elif action == Action.SELECT_2:
            state.select = 1
        elif action == Action.SELECT_3:
            state.select = 2
        else:
            raise ValueError("Unknown action: {}".format(action))

        return state

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return [self.state.__hash__()]  # needs to return a list

