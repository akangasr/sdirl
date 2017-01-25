
import numpy as np

from sdirl.model import RLModel, ELFIModel
from sdirl.rl.simulator import RLSimulator
from sdirl.gridworldmodel.mdp import GridWorldEnvironment, GridWorldTask
from sdirl.gridworldmodel.mdp import State
from sdirl.gridworldmodel.mdp import InitialStateUniformlyAtEdge, InitialStateUniformlyAnywhere

import elfi
from elfi.bo.gpy_model import GPyModel
import GPy

import logging
logger = logging.getLogger(__name__)


class Observation():
    """ Summary observation: start state of path and path length
    """
    def __init__(self, path=None, start_state=None, path_len=None):
        if path is not None:
            self.start_state = path.transitions[0].prev_state
            self.path_len = len(path)
        else:
            self.start_state = start_state
            self.path_len = path_len

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.start_state, self.path_len).__hash__()

    def __repr__(self):
        return "O({},{})".format(self.start_state, self.path_len)

    def __str__(self):
        return self.__repr__()


class GridWorldModel(RLModel, ELFIModel):
    """ Grid world model

    Parameters
    ----------
    variable_names : names of state features
    grid size : int
    world_seed : int
    """
    def __init__(self,
            variable_names,
            grid_size=11,
            step_penalty=0.1,
            prob_rnd_move=0.1,
            world_seed=0,
            n_training_episodes=1000,
            n_episodes_per_epoch=10,
            n_simulation_episodes=100,
            initial_state="edge",
            verbose=False):
        super(GridWorldModel, self).__init__(variable_names, verbose)

        self.initial_state = initial_state
        if self.initial_state == "edge":
            self.initial_state_generator = InitialStateUniformlyAtEdge(grid_size)
        elif self.initial_state == "anywhere":
            self.initial_state_generator = InitialStateUniformlyAnywhere(grid_size)

        self.target_state = State(int(grid_size/2), int(grid_size/2))
        self.path_max_len = grid_size*3
        self.env = GridWorldEnvironment(
                    grid_size=grid_size,
                    prob_rnd_move=prob_rnd_move,
                    n_features=len(variable_names),
                    world_seed=world_seed,
                    target_state=self.target_state,
                    initial_state_generator=self.initial_state_generator
                    )
        self.task = GridWorldTask(
                    env=self.env,
                    step_penalty=step_penalty,
                    max_number_of_actions_per_session=self.path_max_len)
        self.rl = RLSimulator(
                    n_training_episodes=n_training_episodes,
                    n_episodes_per_epoch=n_episodes_per_epoch,
                    n_simulation_episodes=n_simulation_episodes,
                    var_names=variable_names,
                    env=self.env,
                    task=self.task)
        if self.verbose is True:
            self.env.print_grid()

    def to_json(self):
        ret = super(GridWorldModel, self).to_json()
        ret["initial_state"] = self.initial_state

    def summarize(self, raw_observations):
        return [self.summary(ses["path"]) for ses in raw_observations["sessions"]]

    @staticmethod
    def summary(path):
        """ Returns a summary observation of the full path
        """
        return Observation(path)

    def _prob_obs(self, obs, path):
        """ Returns the probability that 'path' would generate 'obs'.

        Parameters
        ----------
        obs : tuple (path x0, path y0, path length)
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        """
        # deterministic summary
        if self.summary(path) == obs:
            return 1.0
        return 0.0

    def _fill_path_tree(self, obs):
        """ Recursively fill path tree starting from obs
        """
        if obs not in self._precomp_paths.keys():
            if obs.path_len > 0:
                node = list()
                for transition in self.env.get_transitions(obs.start_state):
                    next_obs = Observation(start_state=transition.next_state,
                                           path_len=obs.path_len-1)
                    node.append((transition, next_obs))
                self._precomp_paths[obs] = node
                for transition, next_obs in node:
                    self._fill_path_tree(next_obs)
            else:
                self._precomp_paths[obs] = tuple()

    def calculate_discrepancy(self, observations, sim_observations):
        features = [self._avg_path_len_by_start(i, observations) for i in range(self.initial_state_generator.n_initial_states)]
        features_sim = [self._avg_path_len_by_start(i, sim_observations) for i in range(self.initial_state_generator.n_initial_states)]
        disc = 0.0
        for f, fs in zip(features, features_sim):
            disc += np.abs(f - fs)
        disc /= len(features)  # scaling
        return disc

    def _avg_path_len_by_start(self, start_id, obs):
        state = self.initial_state_generator.get_initial_state(start_id)
        vals = []
        for o in obs:
            if o.start_state == state:
                vals.append(o.path_len)
        if len(vals) > 0:
            return np.mean(vals)
        return 0.0

    def get_bounds(self):
        ret = []
        for v in self.variable_names:
            ret.append((-1, 0))
        return tuple(ret)

