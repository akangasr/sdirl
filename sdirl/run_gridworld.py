import os
import sys
import time

from sdirl.inference_tasks import *
from sdirl.gridworldmodel.model import GridWorldModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    env = Environment(variant="local")
    seed = 123987123
    cmdargs = sys.argv

    variable_names = ["feature1_value", "feature2_value", "feature3_value"]
    grid_size = 7
    step_penalty = 0.05
    prob_rnd_move = 0.05
    world_seed = 1234
    n_training_episodes = 50000
    n_episodes_per_epoch = 10
    n_simulation_episodes = 100
    initial_state = "edge"
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
        verbose=verbose)
    ground_truth = [0, -0.25, -0.5]

    bolfi_params = BolfiParams(
            n_surrogate_samples = 100,
            batch_size = 4)

    exp = BOLFI_ML_ComparisonExperiment(env,
            seed,
            cmdargs,
            model,
            ground_truth,
            bolfi_params)
    exp.run()

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(file_dir_path, "model.json")
    write_json(model_file, model.to_json())
    exp_file = os.path.join(file_dir_path, "experiment.json")
    write_json(exp_file, exp.to_json())
