import os
import sys

import matplotlib
matplotlib.use('Agg')

from sdirl.experiments import *
from sdirl.model import *
from sdirl.gridworld.model import GridWorldFactory
from sdirl.elfi_utils import BolfiParams

import logging
logger = logging.getLogger(__name__)

def get_model(parameters, ground_truth, grid_size, approximate):
    gwf = GridWorldFactory(parameters,
            grid_size=5,
            step_penalty=0.05,
            prob_rnd_move=0.05,
            world_seed=1234,
            n_training_episodes=10,
            n_episodes_per_epoch=100,
            n_simulation_episodes=100,
            max_sim_episode_len=8,
            ground_truth=ground_truth,
            initial_state="edge",
            grid_type="walls")
    return gwf.get_new_instance(approximate)

def get_bolfi_params(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.sync = True
    params.n_surrogate_samples = 1
    params.batch_size = 1
    params.noise_var = 1.0
    params.kernel_var = 1.0
    params.kernel_scale = 1.0
    params.kernel_class = GPy.kern.RBF
    params.gp_params_optimizer = "scg"
    params.gp_params_max_opt_iters = 10
    params.exploration_rate = 1.0
    params.acq_opt_iterations = 1000
    params.batches_of_init_samples = 1
    params.inference_type = InferenceType.ML
    params.use_store = True
    return params

def run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model, approximate, grid_size):
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results_grid_{}_approx_{}.pdf".format(grid_size, approximate))

    experiment = GroundTruthInferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file))
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment_grid_{}_approx_{}.json".format(grid_size, approximate))
    write_json_file(experiment_file, experiment.to_dict())


if __name__ == "__main__":
    env = Environment(sys.argv)

    parameters = [ModelParameter("feature1_value", bounds=(-0.001, 0))]
    ground_truth = [0.0]

    for grid_size in [3, 5, 7]:
        obs = None
        for approximate in [True, False]:
            model = get_model(parameters, ground_truth, grid_size, approximate)
            if obs is None:
                # use same observation for both inferences
                obs = model.simulator(*ground_truth, random_state=env.random_state)[0]
            model.observation = obs
            bolfi_params = get_bolfi_params(parameters)
            bolfi_params.client = env.client

            run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model, approximate, grid_size)

