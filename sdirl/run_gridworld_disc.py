import os
import sys

import matplotlib
matplotlib.use('Agg')

from sdirl.experiments import *
from sdirl.model import *
from sdirl.gridworld.model import GridWorldFactory
from sdirl.elfi_utils import BolfiParams
from sdirl.rl.simulator import RLParams

import logging
logger = logging.getLogger(__name__)

def get_model(parameters, grid_size, ground_truth, world_seed):
    rl_params = RLParams(
                 n_training_episodes=100000,
                 n_episodes_per_epoch=100,
                 n_simulation_episodes=1000,
                 q_alpha=0.1,
                 q_gamma=0.98,
                 exp_epsilon=0.1,
                 exp_decay=1.0)
    gwf = GridWorldFactory(parameters,
            grid_size=grid_size,
            step_penalty=0.05,
            prob_rnd_move=0.05,
            world_seed=world_seed,
            rl_params=rl_params,
            max_sim_episode_len=999,
            ground_truth=ground_truth,
            initial_state="edge",
            grid_type="walls")
    return gwf.get_new_instance(approximate=True)

def get_bolfi_params(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.sync = False
    params.noise_var = 1.0
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

    experiment = InferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file))
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(experiment_file, experiment.to_dict())


if __name__ == "__main__":
    env = Environment(sys.argv)

    n_features = 2
    #n_features = 3
    #n_features = 4
    #grid_size = 9
    #grid_size = 13
    grid_size = 19
    #grid_size = 23
    #grid_size = 29

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

    model = get_model(parameters, grid_size, ground_truth, world_seed)
    bolfi_params = get_bolfi_params(parameters)
    bolfi_params.n_surrogate_samples = n_samples
    bolfi_params.batch_size = batch
    bolfi_params.client = env.client

    run_ground_truth_inference_experiment(parameters, bolfi_params, ground_truth, model)

