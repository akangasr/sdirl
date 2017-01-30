import math
import numpy as np
from enum import IntEnum

from sdirl.rl.utils import vec_state_to_scalar, InitialStateGenerator
from sdirl.rl.utils import Transition
from sdirl.rl.pybrain_extensions import ParametricLoggingEpisodicTask, ParametricLoggingEnvironment

import logging
logger = logging.getLogger(__name__)

"""An implementation of a simple Grid world model

Definition of the MDP.
"""

class State():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __repr__(self):
        return "({},{})".format(self.x, self.y)

    def __str__(self):
        return self.__repr__()


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class InitialStateUniformlyAtEdge(InitialStateGenerator):
    """ Returns a state randomly from the edge of the grid
    """

    def __init__(self, grid_size):
        super(InitialStateUniformlyAtEdge, self).__init__(grid_size)
        self.n_initial_states = (self.grid_size - 1) * 4

    def get_initial_state(self, id_number):
        x = 0
        y = 0
        lim = self.grid_size - 1
        if id_number < 0:
            raise ValueError("Id was {}, expected at least 0"
                    .format(id_number))
        elif id_number < lim:
            x = id_number
        elif id_number < lim * 2:
            x = lim
            y = id_number - lim
        elif id_number < lim * 3:
            x = 3 * lim - id_number
            y = lim
        elif id_number < lim * 4:
            y = lim * 4 - id_number
        else:
            raise ValueError("Id was {}, expected less than {}"
                    .format(id_number, lim*4))
        return State(x, y)


class InitialStateUniformlyAnywhere(InitialStateGenerator):
    """ Returns a state randomly from the grid, except for the center
    """

    def __init__(self, grid_size):
        super(InitialStateUniformlyAnywhere, self).__init__(grid_size)
        self.n_initial_states = self.grid_size ** 2 - 1

    def get_initial_state(self, id_number):
        if id_number < 0:
            raise ValueError("Id was {}, expected at least 0"
                    .format(id_number))
        if id_number >= self.n_initial_states:
            raise ValueError("Id was {}, expected less than {}"
                    .format(id_number, self.n_initial_states))
        x = id_number % self.grid_size
        y = int(id_number / self.grid_size)
        if x == int(self.grid_size/2) and y == int(self.grid_size/2):
            x = self.grid_size - 1
            y = self.grid_size - 1
        return State(x, y)


class GridWorldTask(ParametricLoggingEpisodicTask):

    def __init__(self, env, max_number_of_actions_per_session, step_penalty):
        super(GridWorldTask, self).__init__(env)

        self.goal_value = 1.0
        self.env.task = self
        self.max_number_of_actions_per_session = max_number_of_actions_per_session
        self.step_penalty = step_penalty

    def to_dict(self):
        return {
                "max_number_of_actions_per_session": self.max_number_of_actions_per_session,
                "step_penalty": step_penalty
                }

    def setup(self, variables):
        self.v = variables

    def calculate_value(self, state):
        features = self.env.get_state_features(state)
        value = features[0] * self.goal_value
        for i in range(1, len(features)):
            value += features[i] * self.v["feature{}_value".format(i)]
        return value

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        return self.calculate_value(self.env.state) - self.step_penalty

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.state == self.env.target_state:
            return True
        return False


class GridWorldEnvironment(ParametricLoggingEnvironment):
    """ Grid environment

    Parameters
    ----------
    grid_size : int
        width and height of square grid
    prob_rnd_move : float
        probability of agent moving randomly during action
    world_seed : int
        seed for generating world features
    target_state : State
        target location
    """

    def __init__(self,
            grid_size=3,
            prob_rnd_move=0.1,
            n_features=2,
            world_seed=0,
            target_state=None,
            initial_state_generator=None
            ):
        super(GridWorldEnvironment, self).__init__()
        self.task = None # set by Task

        assert grid_size > 0, grid_size
        assert 0 <= prob_rnd_move <= 1, prob_rnd_move
        assert target_state is not None, target_state

        self.grid_size = grid_size
        self.prob_rnd_move = prob_rnd_move
        self.n_features = n_features
        self.world_seed = world_seed
        self.target_state = target_state
        self.initial_state_generator = initial_state_generator
        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.grid = self._generate_grid()

        # pybrain variables
        self.discreteStates = True
        self.outdim = 1
        self.indim = 1
        self.discreteActions = True
        self.numActions = len(self.actions)

    def to_dict(self):
        return {
                "grid_size": self.grid_size,
                "prob_rnd_move": self.prob_rnd_move,
                "n_features": self.n_features,
                "world_seed": self.world_seed,
                "target_state": self.target_state,
                }

    def _generate_grid(self):
        """ Returns the grid
        """
        rs = np.random.RandomState(self.world_seed)
        grid = dict()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = State(x, y)
                features = list()
                for k in range(self.n_features+1):
                    if k == 0:
                        # first feature is unique to goal state
                        if state == self.target_state:
                            features.append(1)
                        else:
                            features.append(0)
                    else:
                        if state == self.target_state:
                            # goal state doesn't have other features
                            features.append(0)
                        elif rs.uniform(0,1) < 0.4:
                            features.append(1)
                        else:
                            features.append(0)
                grid[state] = features
        return grid

    def print_grid(self):
        """ Visualizes grid values
        """
        for x in reversed(range(self.grid_size)):
            s = list()
            for y in range(self.grid_size):
                s.append("{}".format(self.grid[State(x, y)]))
            logger.info("".join(s))

    def print_policy(self, policy):
        """ Visualizes policy
        """
        for x in reversed(range(self.grid_size)):
            s = list()
            for y in range(self.grid_size):
                state = State(x, y)
                s.append("[{:+1.2f} ".format(float(self.task.calculate_value(state))))
                if policy(state, Action.UP) > 0.5:
                    s.append("^]")
                elif policy(state, Action.DOWN) > 0.5:
                    s.append("v]")
                elif policy(state, Action.LEFT) > 0.5:
                    s.append("<]")
                elif policy(state, Action.RIGHT) > 0.5:
                    s.append(">]")
                else:
                    s.append("?]")
            logger.info("".join(s))

    def reset(self):
        """ Called by the library to reset the state """
        self.start_loc_id = self.initial_state_generator.get_random_initial_state_id(self.random_state)
        self.state = self.initial_state_generator.get_initial_state(self.start_loc_id)
        self.n_actions = 0
        self._start_log_for_new_session()

    def in_goal(self):
        return self.state == self.target_state

    def performAction(self, action):
        """ Changes the state of the environment based on agent action """
        self.action = Action(int(action[0]))
        self.prev_state = self.state
        self.state = self.do_transition(self.state, self.action)
        self.n_actions += 1
        self._log_transition()

    def do_transition(self, state, action):
        """ Returns next_state ~ p(next_state | state, action)
        """
        if self.random_state.rand() < self.prob_rnd_move:
            action = self.random_state.choice(self.actions)
        return self._apply_action(state, action)

    def _apply_action(self, state, act):
        """ Applies action to state, deterministic, returns next state
        """
        if act == Action.UP:
            s = State(state.x+1, state.y)
        elif act == Action.DOWN:
            s = State(state.x-1, state.y)
        elif act == Action.LEFT:
            s = State(state.x, state.y-1)
        elif act == Action.RIGHT:
            s = State(state.x, state.y+1)
        else:
            raise ValueError("Unknown action: {}".format(act))
        return self.restrict_state(s)

    def restrict_state(self, state):
        """ Return state that is restricted to possible values of x and y
        """
        return State(x = min(self.grid_size-1, max(0, state.x)),
                     y = min(self.grid_size-1, max(0, state.y)))

    def get_state_features(self, state):
        """ Returns a list of the features of 'state'
        """
        return self.grid[state]

    def get_transitions(self, state):
        """ Returns set of transitions that could be taken from 'state'
        """
        assert type(state) is State
        ret = set()
        states = self._get_neighboring_states(state)
        # Any action can be taken and lead to any neighboring state
        for s in states:
            for a in self.actions:
                ret.add(Transition(state, a, s))
        return ret

    def _get_neighboring_states(self, state):
        """ Returns a set of possible neighboring states
        """
        return set([self.restrict_state(State(state.x+1, state.y)),
                self.restrict_state(State(state.x-1, state.y)),
                self.restrict_state(State(state.x, state.y+1)),
                self.restrict_state(State(state.x, state.y-1))])

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        return [self.state.__hash__()]  # needs to return a list

    def transition_prob(self, transition):
        """ returns p(next_state | state, action)
        """
        possible_next_states = self._get_neighboring_states(transition.prev_state)
        n_possible_next_states = len(possible_next_states)
        if transition.next_state not in possible_next_states:
            return 0.0
        intended_state = self._apply_action(transition.prev_state, transition.action)
        if transition.next_state == intended_state:
            if n_possible_next_states == 4:
                # moving freely or against wall
                return 1.0 - 3*self.prob_rnd_move/4.0
            elif transition.prev_state != transition.next_state:
                # moving from corner
                return 1.0 - 3*self.prob_rnd_move/4.0
            else:
                # moving against corner
                return 1.0 - self.prob_rnd_move/2.0
        else:
            if n_possible_next_states == 4:
                # random move freely or against wall
                return self.prob_rnd_move/4.0
            elif transition.prev_state != transition.next_state:
                # random move from corner
                return self.prob_rnd_move/4.0
            else:
                # random move against corner
                return self.prob_rnd_move/2.0

