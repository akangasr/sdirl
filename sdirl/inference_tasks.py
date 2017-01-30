import os
import sys
import numpy as np
import random
import json
import GPy
import time

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

class BolfiPosteriorUtility():
    """ Allows BolfiPosterior to be serialized, deserialized and constructed from a model
    """

    @staticmethod
    def from_model(model):
        return elfi.posteriors.BolfiPosterior(model, None)

    @staticmethod
    def to_dict(posterior):
        # hack
        model = posterior.model
        data = {
            "X_params": model.gp.X.tolist(),
            "Y_disc": model.gp.Y.tolist(),
            "kernel_class": model.kernel_class.__name__,  # assume set
            "kernel_var": model.kernel_var,  # assume set
            "kernel_scale": model.kernel_scale,  # assume set
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
        return data

    @staticmethod
    def from_dict(data):
        # hack
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
        for x, y in zip(X, Y):
            # we hope here that the gp optimization finds the same location
            model.update(x[None, :], y[None, :])
        posterior = BolfiPosterior(model, data["threshold"])
        posterior.ML = np.atleast_1d(data["ML"])
        posterior.ML_val = data["ML_val"]
        posterior.MAP = np.atleast_1d(data["MAP"])
        posterior.MAP_val = data["MAP_val"]
        return posterior


class BolfiParams():
    """ Encapsulates BOLFI parameters
    """
    def __init__(self, n_surrogate_samples, batch_size):
        self.n_surrogate_samples = n_surrogate_samples
        self.batch_size = batch_size


class BOLFI_ML_ComparisonExperiment():
    """ Experiment where we have two competing methods for inferring the ML estimate of a model:
        approximate and exact likelihood maximization

    Parameters
    ----------
    seed : int
        given this, the experiment sould be deterministic
    cmdargs : list
        command line arguments
    model : sdirl.model.Model
    ground_truth : list
        ground truth parameter values
    bolfi_params : BolfiParams
    """
    def __init__(self, env, seed, cmdargs, model, ground_truth, bolfi_params):
        self.env = env
        self.seed = seed
        self.cmdargs = cmdargs
        self.rs, self.client = self.env.setup(self.seed, self.cmdargs)
        self.model = model
        self.ground_truth = ground_truth
        self.bolfi_params = bolfi_params
        self.obs = list()
        self.results = {
                    "posteriors_disc": list(),
                    "errors_disc": list(),
                    "duration_disc": -1,
                    "posteriors_logl": list(),
                    "errors_logl": list(),
                    "duration_logl": -1,
                    }

    def run(self):
        self._generate_observations()
        start_time = time.time()
        posteriors_disc, errors_disc = self._run_inference(approximate=True)
        mid_time = time.time()
        posteriors_logl, errors_logl = self._run_inference(approximate=False)
        end_time = time.time()
        self.results = {
                "posteriors_disc": posteriors_disc,
                "errors_disc": errors_disc,
                "duration_disc": mid_time - start_time,
                "posteriors_logl": posteriors_disc,
                "errors_logl": errors_logl,
                "duration_logl": end_time - mid_time,
                }

    def _generate_observations(self):
        """ Generate observations for the inference task
        """
        logger.info("Simulating observations at {} ..".format(self.ground_truth))
        self.obs = self.model.simulate_observations(self.ground_truth, self.rs)

    def _construct_BOLFI(self, model, approximate):
        """ Constructs bolfi inference, returns it and store
        """
        wrapper = BOLFIModelWrapper(model, self.obs, approximate=approximate)
        bolfi, store = wrapper.construct_BOLFI(n_surrogate_samples=self.bolfi_params.n_surrogate_samples,
                                               batch_size=self.bolfi_params.batch_size,
                                               client=self.client)
        return bolfi, store

    def _calculate_errors(self, posterior):
        """ Calculates error in ML estimate
        """
        return np.linalg.norm(np.array(self.ground_truth) - posterior.ML, ord=2)

    def _run_inference(self, approximate):
        logger.info("Inferring ML (approximate={})..".format(approximate))
        bolfi, store = self._construct_BOLFI(self.model, approximate=approximate)
        bolfi.infer()
        posteriors = [BolfiPosteriorUtility.from_model(store.get("BOLFI-model", i)[0])
                for i in range(self.bolfi_params.n_surrogate_samples)]
        errors = [self._calculate_errors(p) for p in posteriors]
        logger.info("Ground truth: {}, ML estimate: {}".format(self.ground_truth, posteriors[-1].ML))
        self._print_errors(errors)
        return posteriors, errors

    def _serialized_results(self):
        return {
                "posteriors_disc": [BolfiPosteriorUtility.to_dict(p) for p in self.results["posteriors_disc"]],
                "errors_disc": self.results["errors_disc"],
                "duration_disc": self.results["duration_disc"],
                "posteriors_logl": [BolfiPosteriorUtility.to_dict(p) for p in self.results["posteriors_logl"]],
                "errors_logl": self.results["errors_logl"],
                "duration_logl": self.results["duration_logl"],
                }

    def _deserialize_results(self, ser_res):
        self.results["posteriors_disc"] = [BolfiPosteriorUtility.from_dict(p) for p in ser_res["posteriors_disc"]]
        self.results["errors_disc"] = ser_res["errors_disc"]
        self.results["duration_disc"] = ser_res["duration_disc"]
        self.results["posteriors_logl"] = [BolfiPosteriorUtility.from_dict(p) for p in ser_res["posteriors_logl"]]
        self.results["errors_logl"] = ser_res["errors_logl"]
        self.results["duration_logl"] = ser_res["duration_logl"]

    def _print_errors(self, errors):
        logger.info("Errors:")
        grid = 30
        lim = max(errors) / float(grid)
        for n in reversed(range(grid)):
            st = ["{: >+5.3f}".format(n*lim)]
            for e in errors:
                if e >= n*lim:
                    st.append("*")
                else:
                    st.append(" ")
            logger.info("".join(st))

    def to_json(self):
        return {
            "seed": self.seed,
            "cmdargs": self.cmdargs,
            "model_class": self.model.__class__.__name__,
            "model": self.model.to_json(),
            "ground_truth": self.ground_truth,
            "bolfi_params": self.bolfi_params.__dict__,
            "obs_class": "" if len(self.obs) == 0 else self.obs[0].__class__.__name__,
            "obs": [o.to_json() for o in self.obs],
            "results": self._serialized_results(),
            }

    @staticmethod
    def from_json(data):
        seed = data["seed"]
        cmdargs = data["cmdargs"]
        model = eval(data["model_class"]).from_json(data["model"])
        ground_truth = data["ground_truth"]
        bolfi_params = BolfiParams()
        bolfi_params.__dict__ = data["bolfi_params"]
        exp = BOLFI_ML_ComparisonExperiment(seed, cmdargs, model, ground_truth, bolfi_params)
        if len(data["obs_class"]) > 0:
            obs_class = eval(data["obs_class"])
            exp.obs = [obs_class.from_json(j) for j in data["obs"]]
        exp._deserialize_results(data["results"])
        return exp


def write_json_file(filename, data):
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
    logger.info("Wrote {}".format(filename))

def read_json_file(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    logger.info("Read {}".format(filename))
    return data


class Environment():
    """ Execution environment setup

    Parameters
    ----------
    variant : string
        "slurm" : cluster environment
        "local" : local workstation
    """
    def __init__(self, variant):
        self.variant = variant
        if self.variant == "slurm":
            logger.info("SLURM environment setup")
            self.setup = Environment.setup_slurm
        if self.variant == "local":
            logger.info("LOCAL environment setup")
            self.setup = Environment.setup_local

    def setup(self, seed, args):
        pass

    @staticmethod
    def setup_local(seed, args):
        """ Default experiment setup for local workstation
        """
        Environment.logging_setup()
        Environment.disable_pybrain_warnings()
        rs = Environment.random_seed_setup(seed)
        client = None
        return rs, client

    @staticmethod
    def setup_slurm(seed, args):
        """ Default experiment setup for slurm cluster
        """
        Environment.logging_setup()
        Environment.setup_matplotlib_backend_no_xenv()
        Environment.disable_pybrain_warnings()
        rs = Environment.random_seed_setup(seed)
        client = Environment.dask_client_setup(args)
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
        if len(args) > 1:
            address = "127.0.0.1:{}".format(int(args[1]))
            logger.info("Dask client at " + address)
            client = Client("127.0.0.1:{}".format(int(args[1])))
            dask.set_options(get=client.get)
        return client


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
        store = elfi.storage.DictListStore()
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
                        store=store,
                        model=gpmodel,
                        acquisition=acquisition,
                        sync=True,
                        bounds=bounds,
                        client=client,
                        n_surrogate_samples=n_surrogate_samples)
        return method, store

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


