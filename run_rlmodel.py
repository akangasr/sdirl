from rlmodel.elfi_inference import do_inference, inference_task
from rlmodel.elfi_inference import load_posterior, store_posterior

import logging
logger = logging.getLogger(__name__)

import warnings

def disable_pybrain_warnings():
    warnings.simplefilter("ignore")

def logging_setup():
    logger.setLevel(logging.INFO)
    logging.getLogger("rlmodel").setLevel(logging.INFO)
    logging.getLogger("elfi").setLevel(logging.DEBUG)
    logging.getLogger("elfi.bo").setLevel(logging.INFO)

if __name__ == "__main__":
    logging_setup()
    disable_pybrain_warnings()
    logger.info("Start")
    variables = ["focus_dur"]
    n_training_episodes = 1000
    n_surrogate_samples = 10
    batch_size = 5
    filename = "out2.json"
    task = inference_task(variables, n_training_episodes)
    posterior = do_inference(task, n_surrogate_samples, batch_size)
    store_posterior(posterior, filename)
    p2 = load_posterior(filename)
    logger.info("End")
