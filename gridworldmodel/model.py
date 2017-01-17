import numpy as np
import scipy as sp
import random

import logging
logger = logging.getLogger("experiment")

from pybrain.rl.environments import Environment, Task, EpisodicTask

from abc4py.utils.utils import *

class GridTask(EpisodicTask):

    max_numbers_of_actions_per_session = 1000

    def __init__(self):
        EpisodicTask.__init__(self, None)

    def reset(self):
        self.env.task = self
        self.env.reset()

    def set_v(self, v):
        self.v = v
        self.env.set_v(v)

    def getReward(self):
        """ Returns the current reward based on the state of the environment"""
        # this function should not change the value of any variables
        return self.env.current_value() - 1.0

    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should not change the value of any variables
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.env.in_goal() is True:
            return True
        return False


class GridEnvironment(Environment):

    grid_size  = 5
    n_features = 1
    simulator  = None
    seed = 1235

    def __init__(self):
        """ Initializes and resets the search environment """
        self.v                 = None
        self.log               = None # set by usermodel

    def set_v(self, v):
        self.v = v

    def reset(self):
        """ Called by the library to reset the state """

        self.n_states        = self.grid_size ** 2
        self.n_state_values  = 2
        self.state_type      = self.simulator.rl_model_param("state_type")
        self.discreteStates  = True
        self.outdim          = 1
        self.indim           = 1
        self.discreteActions = True
        self.numActions      = 4 # one for each direction
        self.actnames = ["down", "right", "up", "left"]

        self.rs = np.random.RandomState(self.seed)
        self.grid = list()
        for i in range(self.grid_size):
            row = list()
            for j in range(self.grid_size):
                state = list()
                for k in range(self.n_features):
                    if k == 0:
                        # goal state feature
                        if i == int(self.grid_size / 2) and j == i:
                            state.append(1.0)
                        else:
                            state.append(0.0)
                    else:
                        if self.rs.uniform(0,1) < 0.3:
                            state.append(self.rs.exponential(2))
                        else:
                            state.append(0.0)
                row.append(state)
            self.grid.append(row)

        self.value = list()
        for i in range(self.grid_size):
            vrow = list()
            for j in range(self.grid_size):
                vrow.append(self.get_value(i, j))
            self.value.append(vrow)

        self.print_debug   = random.random() < 0.0
        self.lastact = -1
        self.lastmove = -1
        self.p_random_move = self.v["p_random_move"]
        self.set_start_loc_anywhere()
        self.n_actions  = 0
        if self.log != None:
            if "session" not in self.log:
                self.log["session"] = 0
                self.log["sessions"] = [dict()]
            else:
                self.log["session"] += 1
                self.log["sessions"].append(dict())
            self.step_data = self.log["sessions"][self.log["session"]]
            self.step_data["grid"]          = self.grid
            self.step_data["start"]         = self.start_id
            self.step_data["observation"]   = list()
            self.step_data["action"]        = list()
            self.step_data["reward"]        = list()
        if self.print_debug is True:
            self.print_state()

    def set_start_loc_anywhere(self):
        self.location_x = int(self.grid_size / 2)
        self.location_y = int(self.grid_size / 2)
        while self.location_x == int(self.grid_size / 2) and self.location_y == self.location_x:
            self.location_x = random.randint(0, self.grid_size - 1)
            self.location_y = random.randint(0, self.grid_size - 1)
        self.start_id = self.location_x * self.grid_size + self.location_y

    def print_state(self):
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
        return self.location_x == int(self.grid_size / 2) and self.location_x == self.location_y

    def current_value(self):
        return self.value[self.location_x][self.location_y]

    def get_value(self, x, y):
        cf = self.grid[x][y]
        ret = 0
        for i in range(len(cf)):
            fname = "feature_%d" % (i)
            if fname in self.v.keys():
                ret += cf[i] * self.v[fname]
            else:
                raise ValueError("%s not set" % (fname))
        return ret

    def performAction(self, action):
        """ Changes the state of the environment based on agent action """
        act = int(action[0])
        self.lastact = act
        if self.log != None:
            self.step_data["observation"].append(self.getSensors())
            self.step_data["action"].append(act)

        if random.random() < self.p_random_move:
            act = random.randint(0,3)
        self.lastmove = act

        if act == 0: # down
            self.location_x = min(self.grid_size - 1, self.location_x + 1)
        elif act == 1: # right
            self.location_y = min(self.grid_size - 1, self.location_y + 1)
        elif act == 2: # up
            self.location_x = max(0, self.location_x - 1)
        elif act == 3: # left
            self.location_y = max(0, self.location_y - 1)
        else:
            raise ValueError("action was %d" % (act))

        if self.log != None:
            self.step_data["reward"].append(self.task.getReward())
        self.n_actions += 1

        if self.print_debug is True:
            self.print_state()

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should not change the value of any variables
        return [int(self.grid_size * self.location_x + self.location_y)] # needs to return a list

