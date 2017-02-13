import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

import matplotlib
matplotlib.use('Agg')

import os
import GPy
import numpy as np

from sdirl.model import ModelParameter
from sdirl.elfi_utils import InferenceTaskFactory, InferenceType, BolfiParams, BolfiFactory
from sdirl.gridworld.model import GridWorldFactory
from sdirl.menumodel.model import MenuSearchFactory
from sdirl.experiments import GroundTruthInferenceExperiment, PlotParams, write_json_file


def get_simple_gridworld_model(parameters, ground_truth, approximate):
    gwf = GridWorldFactory(parameters,
            grid_size=3,
            step_penalty=0.1,
            prob_rnd_move=0.1,
            world_seed=0,
            n_training_episodes=1000,
            n_episodes_per_epoch=10,
            n_simulation_episodes=1,
            max_sim_episode_len=8,
            ground_truth=ground_truth,
            initial_state="edge",
            grid_type="walls")
    return gwf.get_new_instance(approximate)


def get_simple_menusearch_model(parameters, ground_truth):
    msf = MenuSearchFactory(
                 parameters,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=False,
                 n_training_menus=10,
                 n_training_episodes=1000,
                 n_episodes_per_epoch=10,
                 n_simulation_episodes=10,
                 ground_truth=ground_truth)
    return msf.get_new_instance(approximate=True)


def get_basic_bolfi_parameters(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.n_surrogate_samples = 10
    params.batch_size = 2
    params.sync = True
    params.kernel_class = GPy.kern.RBF
    params.noise_var = 1.0
    params.kernel_var = 1.0
    params.kernel_scale = 1.0
    params.gp_params_optimizer = "scg"
    params.gp_params_max_opt_iters = 1
    params.exploration_rate = 1.0
    params.acq_opt_iterations = 1
    params.rbf_scale = 1.0
    params.rbf_amplitude = 1.0
    params.batches_of_init_samples = 1
    params.inference_type = InferenceType.ML
    params.client = None
    params.use_store = True
    return params


def run_test_ground_truth_inference_experiment(parameters, ground_truth, model):
    bolfi_parameters = get_basic_bolfi_parameters(parameters)

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results.pdf")

    experiment = GroundTruthInferenceExperiment(model,
            bolfi_parameters,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file))
    experiment.run()

    if os.path.isfile(results_file):
        os.remove(results_file)
    else:
        assert False

    experiment_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(experiment_file, experiment.to_dict())

    if os.path.isfile(experiment_file):
        os.remove(experiment_file)
    else:
        assert False

@slow  # ~1min
def test_simple_menusearch_ground_truth_inference_experiment():
    parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6))]
    ground_truth = [4.0]
    model = get_simple_menusearch_model(parameters, ground_truth)
    run_test_ground_truth_inference_experiment(parameters, ground_truth, model)


@slow  # ~1min
def test_simple_gridworld_ground_truth_inference_experiment():
    for approximate in [True, False]:
        parameters = [ModelParameter("feature1_value", (-1,0))]
        ground_truth = [-0.5]
        model = get_simple_gridworld_model(parameters, ground_truth, approximate)
        run_test_ground_truth_inference_experiment(parameters, ground_truth, model)

