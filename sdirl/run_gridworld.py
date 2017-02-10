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

def get_model(parameters, ground_truth, world_seed, approximate):
    gwf = GridWorldFactory(parameters,
            grid_size=5,
            step_penalty=0.05,
            prob_rnd_move=0.05,
            world_seed=world_seed,
            n_training_episodes=100000,
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
    params.sync = False
    params.kernel_class = GPy.kern.RBF
    params.gp_params_optimizer = "scg"
    params.gp_params_max_opt_iters = 10
    params.exploration_rate = 1.0
    params.acq_opt_iterations = 1000
    params.batches_of_init_samples = 1
    params.inference_type = InferenceType.ML
    params.use_store = True
    return params

def run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model, approximate):
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results_approx_{}.pdf".format(approximate))

    experiment = GroundTruthInferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file))
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment_approx_{}.json".format(approximate))
    write_json_file(experiment_file, experiment.to_dict())


if __name__ == "__main__":
    env = Environment(sys.argv)

    n_features = 2
    #n_features = 3
    #n_features = 4

    parameters = list()
    n_samples = 0
    batch = 0
    for i in range(1, n_features+1):
        parameters.append(ModelParameter("feature{}_value".format(i), bounds=(-1, 0)))
        n_samples += 100
        batch += 10
    if n_features == 4:
        ground_truth = [-0.2, -0.4, -0.6, -0.8]
    if n_features == 3:
        ground_truth = [-0.25, -0.5, -0.75]
    if n_features == 2:
        ground_truth = [-0.33, -0.67]

    world_seed = env.random_state.randint(1e7)

    obs = None

    for approximate in [True, False]:
        model = get_model(parameters, ground_truth, world_seed, approximate)
        if obs is None:
            # use same observation for both inferences
            obs = model.simulator(*ground_truth, random_state=env.random_state)[0]
        model.observation = obs
        bolfi_params = get_bolfi_params(parameters)
        bolfi_params.n_surrogate_samples = n_samples
        bolfi_params.batch_size = batch
        bolfi_params.client = env.client
        if approximate is True:
            bolfi_params.noise_var = 1.0
            bolfi_params.kernel_var = 1.0
            bolfi_params.kernel_scale = 1.0
            bolfi_params.rbf_scale = 0.05
            bolfi_params.rbf_amplitude = 1.0
        else:
            bolfi_params.noise_var = 0.1
            bolfi_params.kernel_var = 1.0
            bolfi_params.kernel_scale = 10.0
            bolfi_params.rbf_scale = 0.05
            bolfi_params.rbf_amplitude = 10.0

        run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model, approximate)
