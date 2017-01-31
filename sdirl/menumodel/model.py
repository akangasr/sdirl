import numpy as np

from sdirl.menumodel.mdp import SearchEnvironment, SearchTask
from sdirl.rl.simulator import RLSimulator
from sdirl.model import RLModel, ELFIModel

import logging
logger = logging.getLogger(__name__)

class Observation():
    """ Summary observation: task completion in one of the possible scenarios:
        target absent or target present
    """
    def __init__(self, action_durations, target_present):
        self.task_completion_time = sum(action_durations)
        self.target_present = (target_present == True)

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.task_completion_time, self.target_present).__hash__()

    def __repr__(self):
        return "O({},{})".format(self.task_completion_time, self.target_present)

    def __str__(self):
        return self.__repr__()


class MenuSearchModel(RLModel, ELFIModel):
    """ Menu search model.
        From Chen et al. CHI 2016
        Similar as in Kangasraasio et al. CHI 2017
    """

    def __init__(self,
                 variable_names,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=True,
                 p_obs_len_cur=0.95,
                 p_obs_len_adj=0.89,
                 n_training_menus=10000,
                 n_training_episodes=20000000,
                 n_episodes_per_epoch=20,
                 n_simulation_episodes=10000,
                 verbose=True):
        super(MenuSearchModel, self).__init__(variable_names, verbose)
        self.env = SearchEnvironment(
                    menu_type=menu_type,
                    menu_groups=menu_groups,
                    menu_items_per_group=menu_items_per_group,
                    semantic_levels=semantic_levels,
                    gap_between_items=gap_between_items,
                    prop_target_absent=prop_target_absent,
                    length_observations=length_observations,
                    p_obs_len_cur=p_obs_len_cur,
                    p_obs_len_adj=p_obs_len_adj,
                    n_training_menus=n_training_menus)
        self.task = SearchTask(
                    env=self.env,
                    max_number_of_actions_per_session=20)
        self.rl = RLSimulator(
                    n_training_episodes=n_training_episodes,
                    n_episodes_per_epoch=n_episodes_per_epoch,
                    n_simulation_episodes=n_simulation_episodes,
                    var_names=variable_names,
                    env=self.env,
                    task=self.task)

    def evaluate_likelihood(variables, observations, random_state=None):
        raise NotImplementedError("Very difficult to evaluate.")

    def get_observation_dataset(menu_type="Semantic",
                              allowed_users=list(),
                              excluded_users=list(),
                              trials_per_user_present=1,
                              trials_per_user_absent=1):
        """ Returns the Bailly dataset as an observation.
        """
        dataset = BaillyData(menu_type,
                           allowed_users,
                           excluded_users,
                           trials_per_user_present,
                           trials_per_user_absent)
        return dataset.get()

    def print_model(self):
        pass

    def summarize(self, raw_observations):
        return [self.summary(ses["action_duration"], ses["target_present"]) for ses in raw_observations["sessions"]]

    @staticmethod
    def summary(action_durations, target_present):
        """ Returns a summary observation of the full path
        """
        return Observation(action_durations, target_present)

    def calculate_discrepancy(self, observations, sim_observations):
        tct_mean_pre_obs, tct_std_pre_obs = self._tct_mean_std(present=True, obs=observations)
        tct_mean_pre_sim, tct_std_pre_sim = self._tct_mean_std(present=True, obs=sim_observations)
        tct_mean_abs_obs, tct_std_abs_obs = self._tct_mean_std(present=False, obs=observations)
        tct_mean_abs_sim, tct_std_abs_sim = self._tct_mean_std(present=False, obs=sim_observations)
        disc = np.abs(tct_mean_pre_obs - tct_mean_pre_sim) ** 2 \
                + np.abs(tct_std_pre_obs - tct_std_pre_sim) \
                + np.abs(tct_mean_abs_obs - tct_mean_abs_sim) ** 2 \
                + np.abs(tct_std_abs_obs - tct_std_abs_sim)
        disc /= 1000000.0  # scaling
        return disc

    def _tct_mean_std(self, present, obs):
        tct = [o.task_completion_time for o in obs if o.target_present is present]
        return np.mean(tct), np.std(tct)

    def get_bounds(self):
        ret = []
        for v in self.variable_names:
            if v == "focus_duration_100ms":
                ret.append((1, 6))
            else:
                raise ValueError
        return tuple(ret)

