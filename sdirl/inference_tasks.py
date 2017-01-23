import sys
import numpy as np
import random
import json
import GPy

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.methods import BOLFI
from elfi.posteriors import BolfiPosterior

import dask
from distributed import Client

import logging
logger = logging.getLogger(__name__)

import warnings

def default_setup_triton(seed, args):
    """ Default experiment setup for triton
    """
    logging_setup()
    logger.info("Start")
    setup_matplotlib_backend_no_xenv()
    disable_pybrain_warnings()
    rs = random_seed_setup(seed)
    client = dask_client_setup(args)
    return rs, client

def setup_matplotlib_backend_no_xenv():
    """ Set matplotlib backend to Agg, which works when we don't have X-env
    """
    import matplotlib
    matplotlib.use('Agg')

def disable_pybrain_warnings():
    """ Ignore warnings from output
    """
    warnings.simplefilter("ignore")

def logging_setup():
    """ Set logging
    """
    logger.setLevel(logging.INFO)
    model_logger = logging.getLogger("sdirl")
    model_logger.setLevel(logging.INFO)
    model_logger.propagate = False
    elfi_logger = logging.getLogger("elfi")
    elfi_logger.setLevel(logging.INFO)
    elfi_logger.propagate = False
    elfi_methods_logger = logging.getLogger("elfi.methods")
    elfi_methods_logger.setLevel(logging.DEBUG)
    elfi_methods_logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    model_logger.handlers = [ch]
    elfi_logger.handlers = [ch]
    elfi_methods_logger.handlers = [ch]
    logger.handlers = [ch]

def random_seed_setup(seed):
    """ Fix random seeds, return a random value source
    """
    random.seed(seed)
    np.random.seed(random.randint(0, 10e7))
    return np.random.RandomState(random.randint(0, 10e7))

def dask_client_setup(args):
    """ Set up and return a dask client or None
    """
    client = None
    if len(args) == 2:
        address = "127.0.0.1:{}".format(int(args[1]))
        logger.info("Dask client at " + address)
        client = Client("127.0.0.1:{}".format(int(args[1])))
        dask.set_options(get=client.get)
    return client

def ML_inference(seed, model, ground_truth, approximate, n_surrogate_samples, batch_size):
    rs, client = default_setup_triton(seed, sys.argv)
    logger.info("Simulating observations at {} ..".format(ground_truth))
    obs = model.simulate_observations(ground_truth, rs)
    wrapper = BOLFIModelWrapper(model, obs, approximate=approximate)
    bolfi = wrapper.construct_BOLFI(n_surrogate_samples=n_surrogate_samples,
                                    batch_size=batch_size,
                                    client=client)
    logger.info("Inference problem constructed, starting inference..")
    posterior = bolfi.infer()
    logger.info("True values: {}, ML: {}".format(ground_truth, posterior.ML))
    return posterior
    #filename = "out2.json"
    #store_posterior(posterior, None)
    #p2 = load_posterior(filename)

class BOLFIModelWrapper():
    """ Wrapper to allow elfi.BOLFI perform the bayesian optimization for the ML inference

    Parameters
    ----------
    model : sdirl.model.Model
    observations : dataset compatible with observations
    approximate : bool
        if true, will try to evaluate exact likelihood
        if false, will use approximation of likelihood
    """
    def __init__(self, model, observations, approximate=True):
        self.model = model
        self.observations = observations
        self.approximate = approximate

    def construct_BOLFI(self, n_surrogate_samples, batch_size, client):
        itask = InferenceTask()
        bounds = self.model.get_bounds()
        variables = self.model.get_elfi_variables(itask)
        gpmodel = self.model.get_elfi_gpmodel(self.approximate)
        acquisition = None  # default
        model = elfi.Simulator('model',
                        elfi.tools.vectorize(self.simulator),
                        *variables,
                        observed=[["dummy_obs"]],
                        inference_task=itask)
        disc = elfi.Discrepancy('dummy_disc',
                        elfi.tools.vectorize(self.discrepancy),
                        model,
                        inference_task=itask)
        method = BOLFI(distance_node=disc,
                        parameter_nodes=variables,
                        batch_size=batch_size,
                        store=None,
                        model=gpmodel,
                        acquisition=acquisition,
                        sync=True,
                        bounds=bounds,
                        client=client,
                        n_surrogate_samples=n_surrogate_samples)
        return method

    def simulator(self, *variables, random_state=None):
        assert len(variables) == self.model.n_var, variables
        if self.approximate is False:
            ret = self.model.evaluate_loglikelihood(variables, self.observations, random_state)
        else:
            ret = self.model.evaluate_discrepancy(variables, self.observations, random_state)
        return np.atleast_1d(ret)

    def discrepancy(self, data1, data2):
        assert len(data1) == 1, data1
        assert len(data2) == 1, data2
        assert data1[0].shape == (1, ), data1[0]
        assert data2[0].shape == (1, ), data2[0]
        if data2[0] == "dummy_obs":
            val = data1[0]
        elif data1[0] == "dummy_obs":
            val = data2[0]
        else:
            # we should have dummy data
            assert False, (data1, data2)
        if self.approximate is False:
            return -val  # we want maximum likelihood
        else:
            return val  # we want minimum discrepancy

def store_bolfi_posterior(posterior, filename="out.json"):
    # hack
    assert type(posterior) is BolfiPosterior, type(posterior)
    model = posterior.model
    data = {
        "X_params": model.gp.X.tolist(),
        "Y_disc": model.gp.Y.tolist(),
        "kernel_class": model.kernel_class.__name__,
        "kernel_var": float(model.gp.kern.variance.values),
        "kernel_scale": float(model.gp.kern.lengthscale.values),
        "noise_var": model.noise_var,
        "threshold": posterior.threshold,
        "bounds": model.bounds,
        "ML": posterior.ML.tolist(),
        "ML_val": float(posterior.ML_val),
        "MAP": posterior.MAP.tolist(),
        "MAP_val": float(posterior.MAP_val),
        "optimizer": model.optimizer,
        "max_opt_iters": model.max_opt_iters
        }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
    logger.info("Stored compressed posterior to {}".format(filename))

def load_bolfi_posterior(filename="out.json"):
    # hack
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    bounds = data["bounds"]
    kernel_class = eval("GPy.kern.{}".format(data["kernel_class"]))
    kernel_var = data["kernel_var"]
    kernel_scale = data["kernel_scale"]
    noise_var = data["noise_var"]
    optimizer = data["optimizer"]
    max_opt_iters = data["max_opt_iters"]
    model = GPyModel(input_dim=len(bounds),
                    bounds=bounds,
                    kernel_class=kernel_class,
                    kernel_var=kernel_var,
                    kernel_scale=kernel_scale,
                    noise_var=noise_var,
                    optimizer=optimizer,
                    max_opt_iters=max_opt_iters)
    X = np.atleast_2d(data["X_params"])
    Y = np.atleast_2d(data["Y_disc"])
    model._fit_gp(X, Y)
    posterior = BolfiPosterior(model, data["threshold"])
    posterior.ML = np.atleast_1d(data["ML"])
    posterior.ML_val = data["ML_val"]
    posterior.MAP = np.atleast_1d(data["MAP"])
    posterior.MAP_val = data["MAP_val"]
    logger.info("Loaded compressed posterior from {}".format(filename))
    return posterior

