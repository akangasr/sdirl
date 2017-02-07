import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
    )

import numpy as np
import random
import os
import copy

from unittest.mock import Mock

import elfi

from sdirl.inference_tasks import *
from sdirl.model import Model, SimpleGaussianModel

from elfi.posteriors import BolfiPosterior

import logging

def create_simple_model_and_bolfi_inference(approximate):
    elfi_logger = logging.getLogger("elfi.methods")
    elfi_logger.setLevel(logging.DEBUG)
    model = SimpleGaussianModel(["mean"])
    rs = np.random.RandomState(0)
    loc = [0.0]
    obs = model.simulate_observations(loc, rs)
    wrapper = BOLFIModelWrapper(model, obs, approximate=approximate)
    client = elfi.env.client(1,1)
    bolfi_params = BolfiParams(
            n_surrogate_samples=20,
            batch_size=1,
            sync=True,
            exploration_rate=2.0,
            opt_iterations=100)
    bolfi, store = wrapper.construct_BOLFI(bolfi_params=bolfi_params, client=client)
    return bolfi, loc


class TestBOLFI_Experiment():

    def setup_helper(self):
        self.env = Mock(Environment)
        self.env.rs = np.random.RandomState(0)
        self.env.client = None
        self.cmdargs = list()
        self.model = SimpleGaussianModel(["mean"])
        self.ground_truth = [0]
        self.bolfi_params = BolfiParams(50, 1, True)

    def setup_method(self):
        self.setup_helper()
        self.exp = BOLFI_ML_Experiment(self.env,
                self.cmdargs,
                self.model,
                self.ground_truth,
                self.bolfi_params)


class TestBOLFI_ML_SingleExperiment(TestBOLFI_Experiment):

    def setup_method(self):
        self.setup_helper()
        self.exp1 = BOLFI_ML_SingleExperiment(self.env,
                self.model,
                self.ground_truth,
                self.bolfi_params,
                approximate=True)
        self.exp2 = BOLFI_ML_SingleExperiment(self.env,
                self.model,
                self.ground_truth,
                self.bolfi_params,
                approximate=False)

    @slow  # ~10min
    def test_running_generates_reasonable_results(self):
        self.exp1.run()
        assert np.abs(self.exp1.results["results"].posteriors[-1].ML[0] - self.ground_truth[0]) < 0.1
        assert self.exp1.results["results"].errors_L2[-1] < 0.1
        assert self.exp1.results["results"].duration > 1e-3
        self.exp2.run()
        assert np.abs(self.exp2.results["results"].posteriors[-1].ML[0] - self.ground_truth[0]) < 0.1
        assert self.exp2.results["results"].errors_L2[-1] < 0.1
        assert self.exp2.results["results"].duration > 1e-3


class TestBOLFI_ML_ComparisonExperiment(TestBOLFI_Experiment):

    def setup_method(self):
        self.setup_helper()
        self.exp = BOLFI_ML_ComparisonExperiment(self.env,
                self.model,
                self.ground_truth,
                self.bolfi_params)

    @slow  # ~10min
    def test_running_generates_reasonable_results(self):
        self.exp.run()
        assert np.abs(self.exp.results["results_disc"].posteriors[-1].ML[0] - self.ground_truth[0]) < 0.1
        assert np.abs(self.exp.results["results_logl"].posteriors[-1].ML[0] - self.ground_truth[0]) < 0.1
        assert self.exp.results["results_disc"].errors_L2[-1] < 0.1
        assert self.exp.results["results_logl"].errors_L2[-1] < 0.1
        assert self.exp.results["results_disc"].duration > 1e-3
        assert self.exp.results["results_logl"].duration > 1e-3


class TestBOLFIModelWrapper():

    @slow
    def performs_sensible_optimization_with_simple_model(self, approximate):
        np.random.seed(1)
        bolfi, loc = create_simple_model_and_bolfi_inference(approximate)
        posterior = bolfi.infer()
        assert np.abs(posterior.ML[0] - loc[0]) < 0.1

    @slow
    def test_performs_sensible_optimization_with_simple_model_using_likelihood(self):
        self.performs_sensible_optimization_with_simple_model(True)

    @slow
    def test_performs_sensible_optimization_with_simple_model_using_discrepancy(self):
        self.performs_sensible_optimization_with_simple_model(False)


class TestBolfiPosteriorUtility():

    @staticmethod
    def assert_posteriors_equal_within_small_margin(post1, post2):
        for loc in [[0.1], [0.5], [1.0]]:
            logpdf1 = post1.logpdf(loc)
            logpdf2 = post2.logpdf(loc)
            assert np.abs(logpdf1 - logpdf2) < 1e-4
        assert np.abs(post1.threshold - post2.threshold) < 1e-4
        np.testing.assert_array_almost_equal(post1.ML, post2.ML, decimal=2)
        np.testing.assert_array_almost_equal(post1.MAP, post2.MAP, decimal=2)

    @slow  # ~10s
    def test_converting_then_unconverting_does_not_change_posterior(self):
        np.random.seed(1)
        bolfi, l = create_simple_model_and_bolfi_inference(True)
        posterior = bolfi.infer()
        dict_post = BolfiPosteriorUtility.to_dict(posterior)
        posterior2 = BolfiPosteriorUtility.from_dict(dict_post)
        posterior3 = BolfiPosteriorUtility.from_dict(dict_post)
        self.assert_posteriors_equal_within_small_margin(posterior, posterior2)

    @slow  # ~10s
    def test_loading_a_posterior_from_dict_is_consistent(self):
        np.random.seed(1)
        bolfi, l = create_simple_model_and_bolfi_inference(True)
        posterior = bolfi.infer()
        dict_post = BolfiPosteriorUtility.to_dict(posterior)
        posterior2 = BolfiPosteriorUtility.from_dict(dict_post)
        posterior3 = BolfiPosteriorUtility.from_dict(dict_post)
        self.assert_posteriors_equal_within_small_margin(posterior2, posterior3)

    @slow  # ~10s
    def test_making_a_new_posterior_from_existing_model_does_not_change_posterior(self):
        np.random.seed(1)
        bolfi, l = create_simple_model_and_bolfi_inference(True)
        posterior = bolfi.infer()
        posterior2 = BolfiPosteriorUtility.from_model(posterior.model)
        self.assert_posteriors_equal_within_small_margin(posterior, posterior2)

