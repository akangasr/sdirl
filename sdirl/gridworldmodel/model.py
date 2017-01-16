
import numpy as np

from sdirl.model import RLModel
from sdirl.rl.simulator import RLSimulator
from sdirl.rl.utils import PathTreeIterator
from sdirl.gridworldmodel.mdp import GridWorldEnvironment, GridWorldTask
from sdirl.gridworldmodel.mdp import State, Observation
from sdirl.gridworldmodel.mdp import InitialStateUniformlyAtEdge

import elfi
from elfi.bo.gpy_model import GPyModel
import GPy

import logging
logger = logging.getLogger(__name__)

class GridWorldModel(RLModel):
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
            prob_rnd_move=0.1,
            world_seed=0,
            n_training_episodes=1000,
            n_episodes_per_epoch=10,
            n_simulation_episodes=100):
        super(GridWorldModel, self).__init__(variable_names)

        self.initial_state_generator = InitialStateUniformlyAtEdge(grid_size)
        env = GridWorldEnvironment(
                    grid_size=grid_size,
                    prob_rnd_move=prob_rnd_move,
                    n_features=len(variable_names),
                    world_seed=world_seed,
                    target_state=State(int(grid_size/2), int(grid_size/2)),
                    initial_state_generator=self.initial_state_generator
                    )
        task = GridWorldTask(
                    env=env,
                    max_number_of_actions_per_session=grid_size*2)
        self.rl = RLSimulator(
                    n_training_episodes=n_training_episodes,
                    n_episodes_per_epoch=n_episodes_per_epoch,
                    n_simulation_episodes=n_simulation_episodes,
                    var_names=variable_names,
                    env=env,
                    task=task)

        self.precomp_paths = dict()
        self.elfi_variables = list()

    def _fill_path_tree(self, obs):
        """ Recursively fill path tree starting from obs
        """
        if obs not in self._precomp_paths.keys():
            if obs.path_len > 0:
                path = (obs.start_state, )
                for transition in self.env.get_transitions(obs.start_state):
                    path += (Observation(transition.state, obs.path_len-1), )
                self._precomp_paths[obs] = path
            else:
                self._precomp_paths[obs] = (obs.start_state, )

    def _get_optimal_policy(self, variables):
        """ Returns a list of all possible paths that could have generated
            observation 'obs'.
        """
        raise NotImplementedError("Subclass implements")

    def _get_all_paths_for_obs(self, obs):
        """ Returns a tree containing all possible paths that could have generated
            observation 'obs'.
        """
        self._fill_path_tree(obs)
        return PathTreeIterator(obs, self._precomp_paths, obs.path_len)

    def summary(path):
        """ Returns a summary observation of the full path
        """
        return Observation(path.states[0], len(path.states))

    def _prob_obs(self, obs, path):
        """ Returns the probability that 'path' would generate 'obs'.

        Parameters
        ----------
        obs : tuple (path x0, path y0, path length)
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        """
        # deterministic summary
        if summary(path) == obs:
            return 1.0
        return 0.0

    def _prob_path(self, path, policy, transfer):
        """ Returns the probability that 'path' would have been generated given 'policy'.

        Parameters
        ----------
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        policy : callable(state, action) -> p(action | state)
        transfer : callable(state, action, state') -> p(state' | state, action)
        """
        logp = 0
        # assume all start states equally probable
        for i in range(len(path.states)-1):
            act_i_prob = policy(path.states(i), path.actions(i))
            tra_i_prob = transfer(path.states(i), path.actions(i), path.states(i+1))
            logp += np.log(act_i_prob) + np.log(tra_i_prob)
        return np.exp(logp)

    def simulate_observations(self, variables, random_state):
        """ Simulates observations from model with variable values.

        Parameters
        ----------
        variables : list of values, matching variable_names
        random_state : random value source

        Returns
        -------
        Dataset compatible with model
        """
        logger.info("simulating observations at: {}".format(variables))
        return self.rl(*variables, random_state=random_state)

    def calculate_discrepancy(self, observations, sim_observations):
        features = [self._path_len_by_start(i, observations[0]) for i in range(self.initial_state_generator.n_initial_states)]
        features_sim = [self._path_len_by_start(i, sim_observations[0]) for i in range(self.initial_state_generator.n_initial_states)]
        disc = 0.0
        for f, fs in zip(features, features_sim):
            disc += (float(f) - float(fs)) ** 2
        disc /= 1000.0  # scaling
        logger.info("f: {}, f_sim: {}, disc: {}".format(features, features_sim, disc))
        return disc

    @staticmethod
    def _path_len_by_start(start_id, log):
        vals = []
        for ses in log["sessions"]:
            if ses["start"] == start_id:
                vals.append(len(ses["path"]))
        if len(vals) > 0:
            ret = np.mean(vals)
        else:
            ret = 0
        return np.atleast_1d(ret)

    def get_bounds(self):
        ret = []
        for v in self.variable_names:
            ret.append((-0.5, 0))
        return tuple(ret)

    def get_elfi_gpmodel(self, approximate):
        kernel_class = GPy.kern.RBF
        noise_var = 0.05
        model = GPyModel(input_dim=len(self.variable_names),
                        bounds=self.get_bounds(),
                        kernel_class=kernel_class,
                        kernel_var=0.05,
                        kernel_scale=0.1,
                        noise_var=noise_var,
                        optimizer="scg",
                        max_opt_iters=50)
        return model

    def get_elfi_variables(self, inference_task):
        if len(self.elfi_variables) > 0:
            return self.elfi_variables
        bounds = self.get_bounds()
        for v, b in zip(self.variable_names, bounds):
            v = elfi.Prior(v, "uniform", b[0], b[1], inference_task=inference_task)
            self.elfi_variables.append(v)
        return self.elfi_variables
