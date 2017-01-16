import json
import numpy as np
import GPy

from menumodel.observation import BaillyData
from menumodel.discrepancy import Discrepancy
from menumodel.rl_model import SearchTask, SearchEnvironment
from menumodel.rl_base import RLModel
from menumodel.summary import feature_extraction

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.methods import BOLFI
from elfi.posteriors import BolfiPosterior

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Create ELFI model, do inference, store results.
"""

def inference_task(inf_vars=list(), n_training_episodes=20000000):
    """Returns a complete Menu model in inference task

    Returns
    -------
    InferenceTask
    """
    dataset = BaillyData(menu_type="Semantic",
                       allowed_users=[],
                       excluded_users=[],
                       trials_per_user_present=1,
                       trials_per_user_absent=1)
    obs = dataset.get()

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

    env = SearchEnvironment(menu_type="semantic",
                            menu_groups=2,
                            menu_items_per_group=4,
                            semantic_levels=3,
                            gap_between_items=0.75,
                            prop_target_absent=0.1,
                            length_observations=True,
                            p_obs_len_cur=0.95,
                            p_obs_len_adj=0.89,
                            n_training_menus=10000)
    task = SearchTask(env=env,
                      max_number_of_actions_per_session=20)
    rl = RLModel(n_training_episodes=n_training_episodes,
                 n_episodes_per_epoch=10,
                 n_simulation_episodes=10000,
                 var_names=var_names,
                 env=env,
                 task=task)
    model = elfi.Simulator('RL', elfi.tools.vectorize(rl), *variables, observed=obs, inference_task=itask)

    features = elfi.Summary('features', feature_extraction, model, inference_task=itask)

    used_discrepancy_features = {
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
    logger.info("Used discrepancy features: {}".format([v for v in used_discrepancy_features.keys() if used_discrepancy_features[v] is True]))
    discrepancy = Discrepancy(used_discrepancy_features=used_discrepancy_features)
    disc = elfi.Discrepancy('discrepancy', elfi.tools.vectorize(discrepancy), features, inference_task=itask)

    itask.parameters = variables
    itask.bounds = bounds
    logger.info("ELFI model done")
    return itask

def do_inference(inference_task, n_surrogate_samples=10, batch_size=1):
    acquisition = None  #default
    bounds = inference_task.bounds
    client = None  #default
    store = None # elfi.storage.DictListStore()
    kernel_class = "GPy.kern.RBF"
    noise_var = 0.05
    model = GPyModel(input_dim=len(bounds),
                    bounds=bounds,
                    kernel_class=eval(kernel_class),
                    kernel_var=0.05,
                    kernel_scale=0.1,
                    noise_var=noise_var,
                    optimizer="scg",
                    max_opt_iters=50)
    method = BOLFI(distance_node=inference_task.discrepancy,
                    parameter_nodes=inference_task.parameters,
                    batch_size=batch_size,
                    store=None,
                    model=model,
                    acquisition=acquisition,
                    sync=False,
                    bounds=bounds,
                    client=client,
                    n_surrogate_samples=n_surrogate_samples)
    posterior = method.infer()
    return posterior

def store_posterior(posterior, filename="out.json"):
    # hack
    assert type(posterior) is BolfiPosterior, type(posterior)
    model = posterior.model
    data = {
        "X_params": model.gp.X.tolist(),
        "Y_disc": model.gp.Y.tolist(),
        "kernel_class": model.kernel_class.__name__,
        "kernel_var": float(model.gp.kern.variance.values),
        "kernel_scale": float(model.gp.kern.lengthscale.values),
        "noise_var": model.noise_var,
        "threshold": posterior.threshold,
        "bounds": model.bounds,
        "ML": posterior.ML.tolist(),
        "ML_val": float(posterior.ML_val),
        "MAP": posterior.MAP.tolist(),
        "MAP_val": float(posterior.MAP_val),
        "optimizer": model.optimizer,
        "max_opt_iters": model.max_opt_iters
        }
    if filename is not None:
       f = open(filename, "w")
       json.dump(data, f)
       f.close()
    else:
        print("-----POSTERIOR-----")
        print(json.dumps(data))
        print("-------------------")
    logger.info("Stored compressed posterior to {}".format(filename))

def load_posterior(filename="out.json"):
    # hack
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    bounds = data["bounds"]
    kernel_class = eval("GPy.kern.{}".format(data["kernel_class"]))
    kernel_var = data["kernel_var"]
    kernel_scale = data["kernel_scale"]
    noise_var = data["noise_var"]
    optimizer = data["optimizer"]
    max_opt_iters = data["max_opt_iters"]
    model = GPyModel(input_dim=len(bounds),
                    bounds=bounds,
                    optimizer=optimizer,
                    max_opt_iters=max_opt_iters)
    model.set_kernel(kernel_class=kernel_class, kernel_var=kernel_var, kernel_scale=kernel_scale)
    X = np.atleast_2d(data["X_params"])
    Y = np.atleast_2d(data["Y_disc"])
    model._fit_gp(X, Y)
    posterior = BolfiPosterior(model, data["threshold"])
    posterior.ML = np.atleast_1d(data["ML"])
    posterior.ML_val = data["ML_val"]
    posterior.MAP = np.atleast_1d(data["MAP"])
    posterior.MAP_val = data["MAP_val"]
    return posterior
