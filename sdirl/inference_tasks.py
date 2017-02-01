import os
import sys
import numpy as np
import random
import json
import GPy
import time
from copy import deepcopy

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.methods import BOLFI
from elfi.posteriors import BolfiPosterior

import dask
from distributed import Client

from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

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
            "X_params": model.gp.X.tolist() if model.gp is not None else [],
            "Y_disc": model.gp.Y.tolist() if model.gp is not None else [],
            "kernel_class": model.kernel.__class__.__name__,
            "kernel_var": float(model.kernel.variance),
            "kernel_scale": float(model.kernel.lengthscale),
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

class BOLFI_Experiment():
    """ Base class for BOLFI experiments

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
        self.results = dict()

    def run(self):
        """ Runs the experiment
        """
        raise NotImplementedError("Subclass implements")

    def _generate_observations(self):
        """ Generate observations for the inference task
        """
        logger.info("Simulating observations at {} ..".format(self.ground_truth))
        self.obs = deepcopy(self.model).simulate_observations(self.ground_truth, self.rs)

    def _construct_BOLFI(self, model, approximate):
        """ Constructs bolfi inference, returns it and store
        """
        wrapper = BOLFIModelWrapper(deepcopy(model), deepcopy(self.obs), approximate=approximate)
        bolfi, store = wrapper.construct_BOLFI(n_surrogate_samples=self.bolfi_params.n_surrogate_samples,
                                               batch_size=self.bolfi_params.batch_size,
                                               client=self.client)
        return bolfi, store

    def _calculate_errors(self, posterior):
        """ Calculates error in results
        """
        raise NotImplementedError
        return np.linalg.norm(np.array(self.ground_truth) - posterior.ML, ord=2)

    def _run_inference(self, approximate):
        logger.info("Inferring ML (approximate={})..".format(approximate))
        bolfi, store = self._construct_BOLFI(self.model, approximate=approximate)
        start_time = time.time()
        bolfi.infer()
        end_time = time.time()
        posteriors = [BolfiPosteriorUtility.from_model(store.get("BOLFI-model", i)[0])
                for i in range(self.bolfi_params.n_surrogate_samples)]
        errors = [self._calculate_errors(p) for p in posteriors]
        logger.info("Ground truth: {}, ML estimate: {}".format(self.ground_truth, posteriors[-1].ML))
        results = BOLFIExperimentResults(posteriors, errors, end_time - start_time)
        results.print_errors()
        return results

    def _serialized_results(self):
        """ Returns a serializable version of self.results
        """
        ret = dict()
        for k, v in self.results.items():
            ret[k] = v.to_dict()
        return ret

    def _deserialize_results(self, ser_res):
        """ Rewrites self.results from a serializable version
        """
        for k, v in ser_res.items():
            self.results[k] = BOLFIExperimentResults.from_dict(v)

    def print_results(self, pdf, figsize):
        """ Print results into a pdf
        """
        raise NotImplementedError

    def _print_text_page(self, pdf, figsize, text):
        fig = pl.figure(figsize=figsize)
        fig.text(0.02, 0.01, text)
        pdf.savefig()
        pl.close()

    def to_dict(self):
        return {
            "seed": self.seed,
            "cmdargs": self.cmdargs,
            "model_class": self.model.__class__.__name__,
            "model": self.model.to_dict(),
            "ground_truth": self.ground_truth,
            "bolfi_params": self.bolfi_params.__dict__,
            "obs": [str(o) for o in self.obs],
            "results": self._serialized_results(),
            }

    @staticmethod
    def from_dict(data):
        seed = data["seed"]
        cmdargs = data["cmdargs"]
        model = eval(data["model_class"]).from_dict(data["model"])
        ground_truth = data["ground_truth"]
        bolfi_params = BolfiParams()
        bolfi_params.__dict__ = data["bolfi_params"]
        exp = BOLFI_ML_ComparisonExperiment(seed, cmdargs, model, ground_truth, bolfi_params)
        exp.obs = data["obs"]  # TODO: proper serialization of results
        exp._deserialize_results(data["results"])
        return exp


class BOLFI_ML_Experiment(BOLFI_Experiment):

    def _calculate_errors(self, posterior):
        """ Calculates L2 ML error
        """
        return np.linalg.norm(np.array(self.ground_truth) - posterior.ML, ord=2)


class BOLFIExperimentResults():

    def __init__(self, posteriors, errors, duration):
        self.posteriors = posteriors
        self.errors = errors
        self.duration = duration

    def print_errors(self, grid=30):
        logger.info("Errors:")
        lim = max(self.errors) / float(grid)
        for n in reversed(range(grid)):
            st = ["{: >+5.3f}".format(n*lim)]
            for e in self.errors:
                if e >= n*lim:
                    st.append("*")
                else:
                    st.append(" ")
            logger.info("".join(st))

    def plot_info(self, pdf, figsize):
        fig = pl.figure(figsize=figsize)
        fig.text(0.02, 0.01, "Duration: {} seconds".format(self.duration))
        pdf.savefig()
        pl.close()

    def plot_errors(self, pdf, figsize):
        fig = pl.figure(figsize=figsize)
        t = range(len(self.errors))
        pl.plot(t, self.errors)
        pl.xlabel("BO samples")
        pl.ylabel("L2 error in ML estimate")
        pl.title("Reduction in error over time")
        pdf.savefig()
        pl.close()

    def plot_posteriors(self, pdf, figsize):
        for posterior in self.posteriors:
            self.plot_posterior(pdf, figsize, posterior)

    def plot_posterior(self, pdf, figsize, posterior):
        if posterior.model.gp is None:
            return
        if posterior.model.input_dim not in [1, 2]:
            return
        fig = pl.figure(figsize=figsize)
        posterior.model.gp.plot()
        pdf.savefig()
        pl.close()

    def plot_pdf(self, pdf, figsize):
        self.plot_info(pdf, figsize)
        self.plot_errors(pdf, figsize)
        self.plot_posteriors(pdf, figsize)

    def to_dict(self):
        return {
                "posteriors": [BolfiPosteriorUtility.to_dict(p) for p in self.posteriors],
                "errors": self.errors,
                "duration": self.duration,
                }

    @staticmethod
    def from_dict(d):
        posteriors = [BolfiPosteriorUtility.from_dict(p) for p in d["posteriors"]]
        errors = d["errors"]
        duration = d["duration"]
        return BOLFIExperimentResults(posteriors, errors, duration)


class BOLFI_ML_SingleExperiment(BOLFI_ML_Experiment):
    """ Experiment where we have one method for inferring the ML estimate of a model:
        approximate or exact likelihood maximization

    Parameters
    ----------
    approximate : bool
        True: discrepancy based inference
        False: likelihood based inference
    """
    def __init__(self, env, seed, cmdargs, model, ground_truth, bolfi_params, approximate):
        super(BOLFI_ML_Experiment, self).__init__(env, seed, cmdargs, model, ground_truth, bolfi_params)
        self.approximate = approximate
        logger.info("BOLFI ML EXPERIMENT WITH APPROXIMATE={}".format(self.approximate))

    def run(self):
        self._generate_observations()
        results = self._run_inference(approximate=self.approximate)
        self.results = {
                "results": results,
                }

    def print_results(self, pdf, figsize):
        if self.approximate is True:
            self._print_text_page(pdf, figsize, "Approximate inference")
        else:
            self._print_text_page(pdf, figsize, "Exact inference")
        self.results["results"].plot_pdf(pdf, figsize)


class BOLFI_ML_ComparisonExperiment(BOLFI_ML_Experiment):
    """ Experiment where we have two competing methods for inferring the ML estimate of a model
        using the exact same set of observations: approximate and exact likelihood maximization
    """

    def __init__(self, env, seed, cmdargs, model, ground_truth, bolfi_params):
        super(BOLFI_ML_ComparisonExperiment, self).__init__(env, seed, cmdargs, model, ground_truth, bolfi_params)
        logger.info("BOLFI ML COMPARISON EXPERIMENT")

    def run(self):
        self._generate_observations()
        results_disc = self._run_inference(approximate=True)
        results_logl = self._run_inference(approximate=False)
        self.results = {
                "results_disc": results_disc,
                "results_logl": results_logl,
                }

    def print_results(self, pdf, figsize):
        self._print_text_page(pdf, figsize, "Approximate inference")
        self.results["results_disc"].plot_pdf(pdf, figsize)
        self._print_text_page(pdf, figsize, "Exact inference")
        self.results["results_logl"].plot_pdf(pdf, figsize)


def write_report_file(filename, experiment):
    figsize = (8.27, 11.69)  # A4 portrait
    with PdfPages(filename) as pdf:
        experiment.print_results(pdf, figsize)
    logger.info("Wrote {}".format(filename))

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
        "distributed" : separate dask.distributed scheduler
        "local" : default scheduler
    """
    def __init__(self, variant):
        self.variant = variant
        Environment.logging_setup()
        Environment.disable_pybrain_warnings()
        if self.variant == "distributed":
            logger.info("DISTRIBUTED environment setup")
            self.setup = Environment.setup_distributed
        if self.variant == "local":
            logger.info("LOCAL environment setup")
            self.setup = Environment.setup_local

    def setup(self, seed, args):
        pass

    @staticmethod
    def setup_local(seed, args):
        """ Default experiment setup for local
        """
        rs = Environment.random_seed_setup(seed)
        client = None
        return rs, client

    @staticmethod
    def setup_distributed(seed, args):
        """ Default experiment setup for distributed
        """
        rs = Environment.random_seed_setup(seed)
        client = Environment.dask_client_setup(args)
        return rs, client

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
        model_copy = deepcopy(self.model)  # prevents any parallel execution fun-stuff, hopefully
        if self.approximate is False:
            ret = model_copy.evaluate_loglikelihood(variables, self.observations, random_state)
        else:
            ret = model_copy.evaluate_discrepancy(variables, self.observations, random_state)
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


