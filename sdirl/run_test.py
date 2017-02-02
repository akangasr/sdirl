import os
import sys
import time

from sdirl.model import SimpleGaussianModel
from sdirl.inference_tasks import *

import logging
logger = logging.getLogger(__name__)

def run(location):
    env = Environment(variant="local")
    seed = 0
    cmdargs = sys.argv

    model = SimpleGaussianModel(["mean"])
    ground_truth = [0.0]

    bolfi_params = BolfiParams(
            n_surrogate_samples = 20,
            batch_size = 2,
            sync = True)

    exp = BOLFI_ML_SingleExperiment(env,
            seed,
            cmdargs,
            model,
            ground_truth,
            bolfi_params,
            approximate=True)
    exp.run()

    exp_file = os.path.join(location, "experiment.json")
    write_json_file(exp_file, exp.to_dict())
    pdf_file = os.path.join(location, "results.pdf")
    write_report_file(pdf_file, exp)

if __name__ == "__main__":
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    run(file_dir_path)
