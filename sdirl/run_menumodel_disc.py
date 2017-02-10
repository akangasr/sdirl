import os
import sys

import matplotlib
matplotlib.use('Agg')

from sdirl.experiments import *
from sdirl.model import *
from sdirl.menumodel.model import MenuSearchFactory
from sdirl.elfi_utils import BolfiParams

import logging
logger = logging.getLogger(__name__)

def get_model(parameters, ground_truth):
    msf = MenuSearchFactory(
                 parameters,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=False,
                 n_training_menus=50000,
                 n_training_episodes=1000000,
                 n_episodes_per_epoch=100,
                 n_simulation_episodes=10000,
                 ground_truth=ground_truth)
    return msf.get_new_instance(approximate=True)

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

def run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model):
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results.pdf")

    experiment = GroundTruthInferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file),
            error_classes=[L2Error])
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(experiment_file, experiment.to_dict())


if __name__ == "__main__":
    env = Environment(sys.argv)

    parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6))]
    ground_truth = [4.0]

    model = get_model(parameters, ground_truth)
    bolfi_params = get_bolfi_params(parameters)
    bolfi_params.client = env.client

    run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model)

