import os
import sys

import matplotlib
matplotlib.use('Agg')

from sdirl.experiments import *
from sdirl.model import *
from sdirl.menumodel.model import MenuSearchFactory, MenuSearch
from sdirl.elfi_utils import BolfiParams
from sdirl.rl.simulator import RLParams

import logging
logger = logging.getLogger(__name__)

def get_model(parameters, ground_truth=None, observation=None):
    rl_params = RLParams(
                 n_training_episodes=100000,
                 n_episodes_per_epoch=100,
                 n_simulation_episodes=10000,
                 q_alpha=0.2,
                 q_gamma=0.98,
                 exp_epsilon=0.1,
                 exp_decay=1.0)
    msf = MenuSearchFactory(
                 parameters,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=False,
                 n_training_menus=5000,
                 rl_params=rl_params,
                 ground_truth=ground_truth,
                 observation=observation)
    return msf.get_new_instance(approximate=True)

def get_dataset():
    return MenuSearch.get_observation_dataset(menu_type="Semantic",
                allowed_users=list(),  # empty = all users
                excluded_users=list(),
                trials_per_user_present=9999,  # all
                trials_per_user_absent=9999)  # all

def get_bolfi_params(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.sync = False
    params.n_surrogate_samples = 100
    params.batch_size = 20
    params.noise_var = 0.05
    params.kernel_var = 1.0
    params.kernel_scale = 1.0
    params.rbf_scale = 0.05
    params.rbf_amplitude = 1.0
    params.kernel_class = GPy.kern.RBF
    params.gp_params_optimizer = "scg"
    params.gp_params_max_opt_iters = 10
    params.exploration_rate = 1.0
    params.acq_opt_iterations = 1000
    params.batches_of_init_samples = 1
    params.inference_type = InferenceType.ML
    params.use_store = True
    return params

def run_inference_experiment(parameters, bolfi_params, model, ground_truth=None):
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results.pdf")

    if ground_truth is not None:
        error_classes=[L2Error]
    else:
        error_classes=[DiscrepancyError]
    experiment = InferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file),
            error_classes=error_classes)
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(experiment_file, experiment.to_dict())

if __name__ == "__main__":
    env = Environment(sys.argv)

    parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6))]
    ground_truth = None

    bolfi_params = get_bolfi_params(parameters)
    bolfi_params.client = env.client

    #ground_truth = [4.0]
    #model = get_model(parameters, ground_truth=ground_truth)

    observation = get_dataset()
    model = get_model(parameters, observation=observation)

    run_inference_experiment(parameters, bolfi_params, model, ground_truth)
