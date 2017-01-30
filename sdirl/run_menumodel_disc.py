import os
import sys
import time

from sdirl.inference_tasks import *
from sdirl.menumodel.model import MenuSearchModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    env = Environment(variant="local")
    seed = 1321412532
    cmdargs = sys.argv

    variable_names = ["focus_duration_100ms"]
    n_training_episodes = 100000
    n_episodes_per_epoch = 10
    n_simulation_episodes = 1000
    verbose = True
    model = MenuSearchModel(variable_names,
        n_training_episodes=n_training_episodes,
        n_episodes_per_epoch=n_episodes_per_epoch,
        n_simulation_episodes=n_simulation_episodes,
        verbose=verbose)
    ground_truth = [0.4]

    bolfi_params = BolfiParams(
            n_surrogate_samples = 100,
            batch_size = 4)

    exp = BOLFI_ML_Experiment(env,
            seed,
            cmdargs,
            model,
            ground_truth,
            bolfi_params,
            approximate=True)
    exp.run()

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(file_dir_path, "model.json")
    write_json(model_file, model.to_dict())
    exp_file = os.path.join(file_dir_path, "experiment.json")
    write_json(exp_file, exp.to_dict())
