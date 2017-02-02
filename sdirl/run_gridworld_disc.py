import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

from sdirl.inference_tasks import *
from sdirl.gridworldmodel.model import GridWorldModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    env = Environment(sys.argv)
    n_features = 1

    if n_features == 3:
        variable_names = ["feature1_value", "feature2_value", "feature3_value"]
        ground_truth = [-0.1, -0.5, -0.9]
    if n_features == 2:
        variable_names = ["feature1_value", "feature2_value"]
        ground_truth = [-0.1, -0.9]
    if n_features == 1:
        variable_names = ["feature1_value"]
        ground_truth = [-0.5]
    grid_size = 3
    step_penalty = 0.1
    prob_rnd_move = 0.05
    world_seed = env.rs.randint(1e7)
    n_training_episodes = 1000000
    n_episodes_per_epoch = 10
    n_simulation_episodes = 1000
    initial_state = "edge"
    grid_type = "walls"
    verbose = True
    model = GridWorldModel(variable_names,
        grid_size=grid_size,
        step_penalty=step_penalty,
        prob_rnd_move=prob_rnd_move,
        world_seed=world_seed,
        n_training_episodes=n_training_episodes,
        n_episodes_per_epoch=n_episodes_per_epoch,
        n_simulation_episodes=n_simulation_episodes,
        initial_state=initial_state,
        grid_type=grid_type,
        verbose=verbose)

    bolfi_params = BolfiParams(
            n_surrogate_samples = 200,
            batch_size = 10,
            sync = False,
            exploration_rate = 2.0,
            opt_iterations = 1000,
            rbf_scale = 0.1,
            rbf_amplitude = 0.5)

    exp = BOLFI_ML_SingleExperiment(env,
            model,
            ground_truth,
            bolfi_params,
            approximate=True)
    exp.run()

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(exp_file, exp.to_dict())
    pdf_file = os.path.join(file_dir_path, "results.pdf")
    write_report_file(pdf_file, exp)
