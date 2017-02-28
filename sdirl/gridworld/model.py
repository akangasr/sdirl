
import numpy as np

from sdirl.environment import Environment
from sdirl.model import *
from sdirl.rl.simulator import RLSimulator, RLParams
from sdirl.gridworld.mdp import *

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

    def copy(self):
        return Observation(self.start_state.copy(), self.path_len)


class GridWorldFactory(SDIRLModelFactory):
    def __init__(self,
            parameters,
            grid_size=11,
            step_penalty=0.1,
            prob_rnd_move=0.1,
            world_seed=0,
            p_grid_feature=0.4,
            rl_params=RLParams(),
            max_sim_episode_len=10000,
            initial_state="edge",
            grid_type="walls",
            observation=None,
            ground_truth=None):

        initial_state = initial_state
        if initial_state == "edge":
            initial_state_generator = InitialStateUniformlyAtEdge(grid_size)
        elif initial_state == "anywhere":
            initial_state_generator = InitialStateUniformlyAnywhere(grid_size)
        else:
            raise ValueError("Unknown initial state type: {}".format(initial_state))

        if grid_type == "uniform":
            grid_generator = UniformGrid(world_seed, p_feature=p_grid_feature)
        elif grid_type == "walls":
            grid_generator = WallsGrid(world_seed, n_walls_per_feature=grid_size)
        else:
            raise ValueError("Unknown grid type: {}".format(grid_type))

        self.max_sim_episode_len = max_sim_episode_len
        goal_state = State(int(grid_size/2), int(grid_size/2))
        path_max_len = grid_size*3
        env = GridWorldEnvironment(
                    grid_size=grid_size,
                    prob_rnd_move=prob_rnd_move,
                    n_features=len(parameters),
                    goal_state=goal_state,
                    initial_state_generator=initial_state_generator,
                    grid_generator=grid_generator)
        task = GridWorldTask(
                    env=env,
                    step_penalty=step_penalty,
                    max_number_of_actions_per_session=path_max_len)
        rl = RLSimulator(
                    rl_params=rl_params,
                    parameters=parameters,
                    env=env,
                    task=task)
        super(GridWorldFactory, self).__init__(name="GridWorld",
                 parameters=parameters,
                 env=env,
                 task=task,
                 rl=rl,
                 goal_state=goal_state,
                 path_max_len=path_max_len,
                 klass=GridWorld,
                 observation=observation,
                 ground_truth=ground_truth)

    def get_new_instance(self, approximate):
        inst = super(GridWorldFactory, self).get_new_instance(approximate)
        inst.max_sim_episode_len = self.max_sim_episode_len
        if approximate is False and inst.observation is None and inst.ground_truth is not None:
            # we use a dummy simulator so this hack is needed
            random_state = Environment.get_instance().random_state
            inst.observation = inst.simulate_observations(*inst.ground_truth, random_state=random_state)[0]
        return inst


class GridWorld(SDIRLModel):
    """ Grid world model

    Built using GridWorldFactory
    """
    def __init__(self):
        super(GridWorld, self).__init__()
        self.max_sim_episode_len = None

    def _filt_obs(self, observations):
        filt_obs = [obs for obs in observations if obs.path_len <= self.max_sim_episode_len]
        if len(filt_obs) < len(observations):
            logger.info("Filtered observations to be at most length {}, left {} out of {}"\
                    .format(self.max_sim_episode_len, len(filt_obs), len(observations)))
        return filt_obs

    def summary_function(self, observations):
        obs = [Observation(ses["path"]) for ses in observations[0].data["sessions"]]
        fobs = self._filt_obs(obs)
        return np.atleast_1d(ObservationDataset(fobs, name="summary"))

    def _prob_obs(self, obs, path):
        """ Returns the probability that 'path' would generate 'obs'.

        Parameters
        ----------
        obs : tuple (path x0, path y0, path length)
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        """
        # deterministic summary
        if Observation(path) == obs:
            return 1.0
        return 0.0

    def _fill_path_tree(self, obs, full_path_len, policy=None):
        """ Recursively fill path tree starting from obs

        Will prune paths that are not feasible:
         * action not possible according to policy
         * goes through the goal state and not end state
         * full path length is less than max, but no way to reach goal state
           with length that is left in obs
        """
        if obs not in self._paths.keys():
            if obs.path_len > 0:
                node = list()
                for transition in self.env.get_transitions(obs.start_state):
                    if policy is not None and policy(transition.prev_state, transition.action) == 0:
                        # impossible action
                        continue
                    next_obs = Observation(start_state=transition.next_state,
                                           path_len=obs.path_len-1)
                    if next_obs.path_len > 0 and next_obs.start_state == self.env.goal_state:
                        # would go through goal state but path does not end there
                        continue
                    if full_path_len < self.path_max_len:
                        # if path is full length we do not know if we reached goal state at the end
                        distance = abs(next_obs.start_state.x - self.env.goal_state.x) \
                                 + abs(next_obs.start_state.y - self.env.goal_state.y)
                        if next_obs.path_len < distance:
                            # impossible to reach goal state with path of this length
                            continue
                    node.append((transition, next_obs))
                if len(node) == 0:
                    # dead end
                    self._paths[obs] = ((None, None),)
                else:
                    self._paths[obs] = node
                    for transition, next_obs in node:
                        self._fill_path_tree(next_obs, full_path_len, policy)
            else:
                self._paths[obs] = tuple()

    def calculate_discrepancy(self, observations, sim_observations):
        features = [self._avg_path_len_by_start(i, observations[0][0].data) for i in range(self.env.initial_state_generator.n_initial_states)]
        features_sim = [self._avg_path_len_by_start(i, sim_observations[0][0].data) for i in range(self.env.initial_state_generator.n_initial_states)]
        disc = 0.0
        for f, fs in zip(features, features_sim):
            disc += np.abs(f - fs)
        disc /= len(features)  # scaling
        return np.atleast_1d([disc])

    def _avg_path_len_by_start(self, start_id, obs):
        state = self.env.initial_state_generator.get_initial_state(start_id)
        vals = []
        for o in obs:
            if o.start_state == state:
                vals.append(o.path_len)
        if len(vals) > 0:
            return np.mean(vals)
        return 0.0

