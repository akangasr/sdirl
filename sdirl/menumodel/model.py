import numpy as np

from sdirl.menumodel.mdp import SearchEnvironment, SearchTask
from sdirl.menumodel.observation import BaillyData
from sdirl.menumodel.features import get_feature_set, plot_features
from sdirl.rl.simulator import RLSimulator, RLParams
from sdirl.model import SDIRLModel, SDIRLModelFactory, ObservationDataset
import elfi

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

    def copy(self):
        return Observation(self.task_completion_time, self.target_present)


class MenuSearchFactory(SDIRLModelFactory):
    def __init__(self,
            parameters,
            menu_type="semantic",
            menu_groups=2,
            menu_items_per_group=4,
            semantic_levels=3,
            gap_between_items=0.75,
            prop_target_absent=0.1,
            length_observations=True,
            p_obs_len_cur=0.95,
            p_obs_len_adj=0.89,
            max_number_of_actions_per_session=20,
            n_training_menus=10000,
            rl_params=RLParams(),
            observation=None,
            ground_truth=None):

        env = SearchEnvironment(
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
        task = SearchTask(
                    env=env,
                    max_number_of_actions_per_session=max_number_of_actions_per_session)
        rl = RLSimulator(
                    rl_params=rl_params,
                    parameters=parameters,
                    env=env,
                    task=task)
        super(MenuSearchFactory, self).__init__(name="MenuSearch",
                 parameters=parameters,
                 env=env,
                 task=task,
                 rl=rl,
                 klass=MenuSearch,
                 observation=observation,
                 ground_truth=ground_truth)


class MenuSearch(SDIRLModel):
    """ Menu search model.
        Chen et al. CHI 2016 and Kangasraasio et al. CHI 2017
    """
    def evaluate_likelihood(variables, observations, random_state=None):
        raise NotImplementedError("Very difficult to evaluate.")

    @staticmethod
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
        return ObservationDataset(dataset.get(), name="Bailly")

    def summary_function(self, obs):
        return ObservationDataset([Observation(ses["action_duration"], ses["target_present"]) for ses in obs.data["sessions"]], name="summary")

    def calculate_discrepancy(self, *simulated, observed=None):
        tct_mean_pre_obs, tct_std_pre_obs = self._tct_mean_std(present=True, obs=simulated[0].data)
        tct_mean_pre_sim, tct_std_pre_sim = self._tct_mean_std(present=True, obs=observed[0].data)
        tct_mean_abs_obs, tct_std_abs_obs = self._tct_mean_std(present=False, obs=simulated[0].data)
        tct_mean_abs_sim, tct_std_abs_sim = self._tct_mean_std(present=False, obs=observed[0].data)
        disc = np.abs(tct_mean_pre_obs - tct_mean_pre_sim) ** 2 \
                + np.abs(tct_std_pre_obs - tct_std_pre_sim) \
                + np.abs(tct_mean_abs_obs - tct_mean_abs_sim) ** 2 \
                + np.abs(tct_std_abs_obs - tct_std_abs_sim)
        disc /= 1000000.0  # scaling
        return np.atleast_1d([disc])

    def _tct_mean_std(self, present, obs):
        tct = [o.task_completion_time for o in obs if o.target_present is present]
        if len(tct) == 0:
            logger.warning("No observations from condition: target present = {}".format(present))
            return 0.0, 0.0
        return np.mean(tct), np.std(tct)

    def plot_obs(self, obs):
        if not isinstance(obs, ObservationDataset):
            print("Can't print observation of type {}".format(type(obs)))
            return
        features = get_feature_set(obs.data)
        plot_features(features)

