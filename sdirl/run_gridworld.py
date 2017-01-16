
import sys
sys.path.append("/scratch/work/akangasr/sdirl/")  # temp fix for triton

from sdirl.inference_tasks import *
from sdirl.gridworldmodel.model import GridWorldModel

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = sys.argv
    model = GridWorldModel(["step_penalty", "feature1_value"],
        grid_size=11,
        prob_rnd_move=0.0,
        world_seed=1234,
        n_training_episodes=500000,
        n_episodes_per_epoch=10,
        n_simulation_episodes=1000)
    ground_truth = [-0.1, -0.1]
    approximate = True
    n_surrogate_samples = 200
    batch_size = 10
    posterior = ML_inference(model, ground_truth, approximate, n_surrogate_samples, batch_size)
    logger.info("End")
