import math
import numpy as np
from enum import Enum

from sdirl.rl.utils import vec_state_to_scalar, InitialStateGenerator

from pybrain.rl.environments import Environment, EpisodicTask

import logging
logger = logging.getLogger(__name__)

"""An implementation of a simple Grid world model

Definition of the MDP.
"""

class Observation():
    def __init__(self, start_state, path_len):
        self.start_state = path_len
        self.path_len = path_len

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.state, self.path_len).__hash__()

class Path():
    def __init__(self, transitions):
        self.transitions = transitions

    def append(self, transition):
        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        ret = list()
        for t in self.transitions:
            ret.append("{};".format(t))
        return "".join(ret)

    def __str__(self):
        return self.__repr__()

class State():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __repr__(self):
        return "({},{})".format(self.x, self.y)

    def __str__(self):
        return self.__repr__()

class Transition():
    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.state.x, self.state.y, self.action).__hash__()

    def __repr__(self):
        return "{}+{}".format(self.state, self.action)

    def __str__(self):
        return self.__repr__()

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GridWorldTask(EpisodicTask):

    def __init__(self, env, max_number_of_actions_per_session):
        EpisodicTask.__init__(self, None)

        self.v = None  # set with setup
        self.goal_value = 1.0

        self.env = env
        self.env.task = self
        self.max_number_of_actions_per_session = max_number_of_actions_per_session

    def setup(self, variables):
        self.v = variables

    def reset(self):
        self.env.reset()

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        features = self.env.get_current_state_features()
        value = self.v["step_penalty"]
        value += features[0] * self.goal_value
        for i in range(1, len(features)):
            value += features[i] * self.v["feature{}_value".format(i)]
        return value

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.state == self.env.target_state:
            return True
        return False

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


class GridWorldEnvironment(Environment):
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
        self.random_state = None # set with setup
        self.log = None # set by RL_model
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

    def setup(self, variables, random_state):
        """ Finishes the initialization
        """
        self.v = variables
        self.random_state = random_state
        self.reset()

    def _generate_grid(self):
        """ Returns the grid
        """
        rs = np.random.RandomState(self.world_seed)
        grid = dict()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = State(x, y)
                features = list()
                for k in range(self.n_features):
                    if k == 0:
                        # first feature is unique to goal state
                        if state == self.target_state:
                            features.append(1.0)
                        else:
                            features.append(0.0)
                    else:
                        if rs.uniform(0,1) < 0.3:
                            features.append(rs.exponential(2))
                        else:
                            features.append(0.0)
                grid[state] = features
        return grid

    def reset(self):
        """ Called by the library to reset the state """
        self.start_loc_id = self.initial_state_generator.get_random_initial_state_id(self.random_state)
        self.state = self.initial_state_generator.get_initial_state(self.start_loc_id)
        self.n_actions = 0
        self._start_log_for_new_session()

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
            self.step_data["grid"] = self.grid
            self.step_data["start"] = self.start_loc_id
            self.step_data["path"] = Path([])
            self.step_data["rewards"] = list()

    def old_print_state(self):
        s = list()
        s.append("step %d: " % (self.n_actions))
        if self.lastact < 0:
            s.append("initial state")
        else:
            s.append("last action: %s, " % (self.actnames[self.lastact]))
            s.append("moved: %s" % (self.actnames[self.lastmove]))
        s.append("\n")
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.location_x == x and self.location_y == y:
                    s.append(" [here] ")
                else:
                    s.append("%7.2f " % (self.value[x][y]))
            s.append("\n")
        print("".join(s))

    def in_goal(self):
        return self.state == self.target_state

    def performAction(self, action):
        """ Changes the state of the environment based on agent action """
        act = Action(int(action[0]))
        if self.log != None:
            self.step_data["path"].append(Transition(self.state, act))

        if self.random_state.rand() < self.prob_rnd_move:
            act = self.random_state.choice(self.actions)

        if act == Action.UP:
            self.state.x += 1
        elif act == Action.DOWN:
            self.state.x -= 1
        elif act == Action.LEFT:
            self.state.y -= 1
        elif act == Action.RIGHT:
            self.state.y += 1
        else:
            raise ValueError("Unknown action: {}".format(act))
        self.state = self.restrict_state(self.state)

        if self.log != None:
            self.step_data["rewards"].append(self.task.getReward())
        self.n_actions += 1

    def restrict_state(self, state):
        """ Return state that is restricted to possible values of x and y
        """
        return State(x = min(self.grid_size-1, max(0, state.x)),
                     y = min(self.grid_size-1, max(0, state.y)))

    def get_current_state_features(self):
        return self.grid[self.state]

    def get_transitions(self, state):
        """ Returns set of transitions that could be taken from 'state'
        """
        ret = set()
        # Can only go to neighboring states
        states = [self._restrict_state(State(state.x+1, state.y)),
                  self._restrict_state(State(state.x-1, state.y)),
                  self._restrict_state(State(state.x, state.y+1)),
                  self._restrict_state(State(state.x, state.y-1))]
        # Any action can be taken and lead to any neighboring state
        for s, a in zip(states, self.actions):
            ret.add(Transition(s, a))
        return ret

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should not change the value of any variables
        return [self.state.__hash__()]  # needs to return a list

