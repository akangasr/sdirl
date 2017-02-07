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
from elfi.bo.acquisition import *
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
    def __init__(self,
            n_surrogate_samples=1,
            batch_size=1,
            sync=True,
            exploration_rate=2.0,
            opt_iterations=100,
            rbf_scale=1.0,
            rbf_amplitude=1.0):
        self.n_surrogate_samples = n_surrogate_samples
        self.batch_size = batch_size
        self.sync = sync
        self.exploration_rate = exploration_rate
        self.opt_iterations = opt_iterations
        self.rbf_scale = rbf_scale
        self.rbf_amplitude = rbf_amplitude

class BOLFI_Experiment():
    """ Base class for BOLFI experiments

    Parameters
    ----------
    model : sdirl.model.Model
    ground_truth : list
        ground truth parameter values
    bolfi_params : BolfiParams
    """
    def __init__(self, env, model, ground_truth, bolfi_params):
        self.env = env
        self.rs = self.env.rs
        self.client = self.env.client
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
        bolfi, store = wrapper.construct_BOLFI(bolfi_params=self.bolfi_params,
                                               client=self.client)
        return bolfi, store

    def _calculate_errors(self, posteriors, duration):
        raise NotImplementedError

    def _run_inference(self, approximate):
        logger.info("Inferring ML (approximate={})..".format(approximate))
        bolfi, store = self._construct_BOLFI(self.model, approximate=approximate)
        start_time = time.time()
        bolfi.infer()
        end_time = time.time()
        posteriors = [BolfiPosteriorUtility.from_model(store.get("BOLFI-model", i)[0])
                for i in range(self.bolfi_params.n_surrogate_samples)]
        results = self._calculate_errors(posteriors, end_time-start_time)
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
            "env": self.env.to_dict(),
            "model_class": self.model.__class__.__name__,
            "model": self.model.to_dict(),
            "ground_truth": self.ground_truth,
            "bolfi_params": self.bolfi_params.__dict__,
            "obs": [str(o) for o in self.obs],
            "results": self._serialized_results(),
            }


class BOLFI_MAP_Experiment(BOLFI_Experiment):

    def _calculate_errors(self, posteriors, duration):
        errors_L2 = [calculate_errors_L2(self.ground_truth, p.MAP) for p in posteriors]
        errors_ord = [calculate_errors_ord(self.ground_truth, p.MAP) for p in posteriors]
        errors_prop = [calculate_errors_prop(self.ground_truth, p.MAP) for p in posteriors]
        logger.info("Ground truth: {}, MAP estimate: {}".format(self.ground_truth, posteriors[-1].MAP))
        results = BOLFIExperimentResults(posteriors, errors_L2, errors_ord, errors_prop, duration)
        logger.info("L2:")
        results.print_errors(errors_L2)
        logger.info("Ordering:")
        results.print_errors(errors_ord)
        logger.info("Proportions:")
        results.print_errors(errors_prop)
        return results


class BOLFI_ML_Experiment(BOLFI_Experiment):

    def _calculate_errors(self, posteriors, duration):
        errors_L2 = [calculate_errors_L2(self.ground_truth, p.ML) for p in posteriors]
        errors_ord = [calculate_errors_ord(self.ground_truth, p.ML) for p in posteriors]
        errors_prop = [calculate_errors_prop(self.ground_truth, p.ML) for p in posteriors]
        logger.info("Ground truth: {}, ML estimate: {}".format(self.ground_truth, posteriors[-1].ML))
        results = BOLFIExperimentResults(posteriors, errors_L2, errors_ord, errors_prop, duration)
        logger.info("L2:")
        results.print_errors(errors_L2)
        logger.info("Ordering:")
        results.print_errors(errors_ord)
        logger.info("Proportions:")
        results.print_errors(errors_prop)
        return results


def calculate_errors_L2(ground_truth, estimates):
    """ Calculates ML error using L2 distance
    """
    return np.linalg.norm(np.array(ground_truth) - estimates, ord=2)

def calculate_errors_ord(ground_truth, estimates):
    """ Calculates ordering-based ML error (hamming distance of ranks)
    """
    order_gt = np.argsort(ground_truth)
    order = np.argsort(estimates)
    return sum([o1 == o2 for o1, o2 in zip(order_gt, order)])

def calculate_errors_prop(ground_truth, estimates):
    """ Calculates proportion-based ML error (L2 distance between vectors of parameter proportions)
    """
    prop_gt = list()
    prop = list()
    for i in range(len(ground_truth)):
        for j in range(i+1, len(ground_truth)):
            prop_gt.append(float(ground_truth[i])/float(ground_truth[j]))
            prop.append(float(estimates[i])/float(estimates[j]))
    return np.linalg.norm(np.array(prop_gt) - np.array(prop), ord=2)


class BOLFIExperimentResults():

    def __init__(self, posteriors, errors_L2, errors_ord, errors_prop, duration):
        self.posteriors = posteriors
        self.errors_L2 = errors_L2
        self.errors_ord = errors_ord
        self.errors_prop = errors_prop
        self.duration = duration

    def print_errors(self, errors, grid=30):
        logger.info("Errors:")
        lim = max(errors) / float(grid)
        for n in reversed(range(grid)):
            st = ["{: >+5.3f}".format(n*lim)]
            for e in errors:
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

    def plot_errors(self, pdf, errors, name, figsize):
        fig = pl.figure(figsize=figsize)
        t = range(len(errors))
        pl.plot(t, errors)
        pl.xlabel("BO samples")
        pl.ylabel("{} error in estimate".format(name))
        pl.title("Reduction in error over time")
        pdf.savefig()
        pl.close()

    def plot_posteriors(self, pdf, figsize):
        for posterior in self.posteriors:
            self.plot_posterior(pdf, figsize, posterior)

    def plot_posterior(self, pdf, figsize, posterior):
        if posterior.model.gp is None:
            return
        fig = pl.figure(figsize=figsize)
        try:
            posterior.model.gp.plot()
        except:
            fig.text(0.02, 0.01, "Could not plot model")
        pdf.savefig()
        pl.close()

    def plot_pdf(self, pdf, figsize):
        self.plot_info(pdf, figsize)
        self.plot_errors(pdf, self.errors_L2, "L2", figsize)
        self.plot_errors(pdf, self.errors_ord, "Order", figsize)
        self.plot_errors(pdf, self.errors_prop, "Proportions", figsize)
        self.plot_posteriors(pdf, figsize)

    def to_dict(self):
        return {
                "posteriors": [BolfiPosteriorUtility.to_dict(p) for p in self.posteriors],
                "errors_L2": [str(v) for v in self.errors_L2],
                "errors_ord": [str(v) for v in self.errors_ord],
                "errors_prop": [str(v) for v in self.errors_prop],
                "duration": self.duration,
                }

    @staticmethod
    def from_dict(d):
        posteriors = [BolfiPosteriorUtility.from_dict(p) for p in d["posteriors"]]
        errors_L2 = d["errors_L2"]
        errors_ord = d["errors_ord"]
        errors_prop = d["errors_prop"]
        duration = d["duration"]
        return BOLFIExperimentResults(posteriors, errors_L2, errors_ord, errors_prop, duration)


class BOLFI_ML_SingleExperiment(BOLFI_ML_Experiment):
    """ Experiment where we have one method for inferring the ML estimate of a model:
        approximate or exact likelihood maximization

    Parameters
    ----------
    approximate : bool
        True: discrepancy based inference
        False: likelihood based inference
    """
    def __init__(self, env, model, ground_truth, bolfi_params, approximate):
        super(BOLFI_ML_Experiment, self).__init__(env, model, ground_truth, bolfi_params)
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


class BOLFI_MAP_SingleExperiment(BOLFI_MAP_Experiment):
    """ Experiment where we have one method for inferring the MAP estimate of a model:
        approximate or exact likelihood maximization

    Parameters
    ----------
    approximate : bool
        True: discrepancy based inference
        False: likelihood based inference
    """
    def __init__(self, env, model, ground_truth, bolfi_params, approximate):
        super(BOLFI_MAP_Experiment, self).__init__(env, model, ground_truth, bolfi_params)
        self.approximate = approximate
        logger.info("BOLFI MAP EXPERIMENT WITH APPROXIMATE={}".format(self.approximate))

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

    def __init__(self, env, model, ground_truth, bolfi_params):
        super(BOLFI_ML_ComparisonExperiment, self).__init__(env, model, ground_truth, bolfi_params)
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
    """
    def __init__(self, args):
        self.logging_setup()
        self.disable_pybrain_warnings()
        self.args = args
        self.rs = self.rng_setup()
        self.client = self.client_setup()

    def to_dict(self):
        return {
                "args": self.args
                }

    def disable_pybrain_warnings(self):
        """ Ignore warnings from output
        """
        warnings.simplefilter("ignore")

    def logging_setup(self):
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

    def rng_setup(self):
        """ Return a random value source
        """
        if len(self.args) > 1:
            seed = self.args[1]
        else:
            seed = 0
        logger.info("Seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(random.randint(0, 10e7))
        return np.random.RandomState(random.randint(0, 10e7))

    def client_setup(self):
        """ Set up and return a dask client or None
        """
        client = None
        if len(self.args) > 2:
            address = "127.0.0.1:{}".format(int(self.args[2]))
            logger.info("Dask client at " + address)
            client = Client(address)
            dask.set_options(get=client.get)
        else:
            logger.info("Default dask client (client=None)")
        return client


class BOLFI_ML_SyncAcquisition(SecondDerivativeNoiseMixin, LCBAcquisition):
    pass

class BOLFI_ML_AsyncAcquisition(RbfAtPendingPointsMixin, LCBAcquisition):
    pass

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

    def construct_BOLFI(self, bolfi_params, client):
        itask = InferenceTask()
        bounds = self.model.get_bounds()
        variables = self.model.get_elfi_variables(itask)
        gpmodel = self.model.get_elfi_gpmodel(self.approximate)
        store = elfi.storage.DictListStore()
        batches_of_init_samples = 1
        n_init_samples =  bolfi_params.batch_size * batches_of_init_samples
        n_acq_samples = bolfi_params.n_surrogate_samples - n_init_samples
        acquisition_init = RandomAcquisition(variables,
                                             n_samples=n_init_samples)
        if bolfi_params.sync is True:
            acquisition = BOLFI_ML_SyncAcquisition(
                    exploration_rate=bolfi_params.exploration_rate,
                    opt_iterations=bolfi_params.opt_iterations,
                    model=gpmodel,
                    n_samples=n_acq_samples)
        else:
            acquisition = BOLFI_ML_AsyncAcquisition(
                    exploration_rate=bolfi_params.exploration_rate,
                    opt_iterations=bolfi_params.opt_iterations,
                    rbf_scale=bolfi_params.rbf_scale,
                    rbf_amplitude=bolfi_params.rbf_amplitude,
                    model=gpmodel,
                    n_samples=n_acq_samples)
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
                        batch_size=bolfi_params.batch_size,
                        store=store,
                        model=gpmodel,
                        acquisition=acquisition_init + acquisition,
                        sync=bolfi_params.sync,
                        bounds=bounds,
                        client=client)
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


