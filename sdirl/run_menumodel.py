
import sys
sys.path.append("/scratch/work/akangasr/sdirl/")  # temp fix for triton

from sdirl.inference_tasks import *
from sdirl.menumodel.model import MenuModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = sys.argv
    model = MenuModel(["focus_duration_100ms"],
        n_training_episodes=20000,
        n_episodes_per_epoch=10,
        n_simulation_episodes=1000)
    ground_truth = [400]
    approximate = True
    n_surrogate_samples = 50
    batch_size = 5
    posterior = ML_inference(model, ground_truth, approximate, n_surrogate_samples, batch_size)
    logger.info("End")
