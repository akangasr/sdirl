import os
import sys
import time

from sdirl.inference_tasks import *
from sdirl.menumodel.model import MenuSearchModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    env = Environment(variant="local")
    seed = 1321412531
    cmdargs = sys.argv

    variable_names = ["focus_duration_100ms"]
    n_training_episodes = 1000000
    n_episodes_per_epoch = 10
    n_simulation_episodes = 10000
    verbose = True
    model = MenuSearchModel(variable_names,
        n_training_episodes=n_training_episodes,
        n_episodes_per_epoch=n_episodes_per_epoch,
        n_simulation_episodes=n_simulation_episodes,
        verbose=verbose)
    ground_truth = [4.0]

    bolfi_params = BolfiParams(
            n_surrogate_samples = 100,
            batch_size = 4)

    exp = BOLFI_ML_SingleExperiment(env,
            seed,
            cmdargs,
            model,
            ground_truth,
            bolfi_params,
            approximate=True)
    exp.run()

    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(exp_file, exp.to_dict())
