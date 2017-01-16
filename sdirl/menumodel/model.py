
from sdirl.model import RLModel

class MenuSearchModel(RLModel):
    """ Menu search model.
        From Chen et al. CHI 2016
        Used in Kangasraasio et al. CHI 2017
    """

    def __init__(self,
                 variable_names,
                 n_training_episodes=20000000,
                 n_episodes_per_epoch=20,
                 n_simulation_episodes=10000):
        super(MenuSearchModel, self).__init__(variable_names)
        env = SearchEnvironment(
                    menu_type="semantic",
                    menu_groups=2,
                    menu_items_per_group=4,
                    semantic_levels=3,
                    gap_between_items=0.75,
                    prop_target_absent=0.1,
                    length_observations=True,
                    p_obs_len_cur=0.95,
                    p_obs_len_adj=0.89,
                    n_training_menus=10000)
        task = SearchTask(
                    env=env,
                    max_number_of_actions_per_session=20)
        self.rl = RLModel(
                    n_training_episodes=n_training_episodes,
                    n_episodes_per_epoch=n_episodes_per_epoch,
                    n_simulation_episodes=n_simulation_episodes,
                    var_names=variable_names,
                    env=env,
                    task=task)
        self.used_discrepancy_features = {
            "00_task_completion_time": True,
            "01_task_completion_time_target_absent": False,
            "02_task_completion_time_target_present": False,
            "03_fixation_duration_target_absent": False,
            "04_fixation_duration_target_present": False,
            "05_saccade_duration_target_absent": False,
            "06_saccade_duration_target_present": False,
            "07_number_of_saccades_target_absent": False,
            "08_number_of_saccades_target_present": False,
            "09_fixation_locations_target_absent": False,
            "10_fixation_locations_target_present": False,
            "11_length_of_skips_target_absent": False,
            "12_length_of_skips_target_present": False,
            "13_location_of_gaze_to_target": False,
            "14_proportion_of_gaze_to_target": False
            }
        logger.info("Used discrepancy features: {}"
                .format([v for v in used_discrepancy_features.keys() if used_discrepancy_features[v] is True]))


    def evaluate_likelihood(variables, observations, random_state=None):
        raise NotImplementedError("Very difficult to evaluate.")

    def simulate_observations(variables, random_state):
        raise NotImplementedError("Subclass implements")

    def calculate_discrepancy(observations1, observations2):
        features1 = feature_extraction(observations1)
        features2 = feature_extraction(observations1)
        discrepancy = Discrepancy(used_discrepancy_features=used_discrepancy_features)

    def get-observation_dataset(menu_type="Semantic",
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


    def old_inference_task():
        """Returns a complete Menu model in inference task

        Returns
        -------
        InferenceTask
        """
        logger.info("Constructing ELFI model..")
        itask = InferenceTask()
        variables = list()
        bounds = list()
        var_names = list()
        for var in inf_vars:
            if var == "focus_dur":
                v = elfi.Prior("focus_duration_100ms", "uniform", 1, 4, inference_task=itask)
                b = (1, 4)
            elif var == "recall_prob":
                v = elfi.Prior("menu_recall_probability", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            elif var == "semantic_obs":
                v = elfi.Prior("prob_obs_adjacent", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            elif var == "selection_delay":
                v = elfi.Prior("selection_delay_s", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            else:
                assert False
            name = v.name
            logger.info("Added variable {}".format(name))
            var_names.append(name)
            variables.append(v)
            bounds.append(b)

        itask.parameters = variables
        itask.bounds = bounds
        logger.info("ELFI model done")
        return itask


