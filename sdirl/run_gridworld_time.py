import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

from sdirl.inference_tasks import *
from sdirl.gridworldmodel.model import GridWorldModel

import logging
logger = logging.getLogger(__name__)


def run(grid_size):
    env = Environment(sys.argv)

    variable_names = ["feature1_value"]
    step_penalty = 0.05
    prob_rnd_move = 0.05
    world_seed = 1234
    n_training_episodes = 1000000
    n_episodes_per_epoch = 10
    n_simulation_episodes = 100
    max_sim_episode_len = 8
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
        max_sim_episode_len=max_sim_episode_len,
        initial_state=initial_state,
        grid_type=grid_type,
        verbose=verbose)
    ground_truth = [0.0]

    bolfi_params = BolfiParams(
            n_surrogate_samples = 1,
            batch_size = 1,
            sync = True)

    exp = BOLFI_ML_ComparisonExperiment(env,
            model,
            ground_truth,
            bolfi_params)
    exp.run()

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_file = os.path.join(file_dir_path, "experiment_{}.json".format(grid_size))
    write_json_file(exp_file, exp.to_dict())
    pdf_file = os.path.join(file_dir_path, "results_{}.pdf".format(grid_size))
    write_report_file(pdf_file, exp)

if __name__ == "__main__":
    run(3)
    run(5)
    run(7)
