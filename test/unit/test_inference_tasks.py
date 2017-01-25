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

from sdirl.inference_tasks import BOLFIModelWrapper
from sdirl.inference_tasks import BolfiPosteriorUtility
from sdirl.inference_tasks import BOLFI_ML_ComparisonExperiment
from sdirl.inference_tasks import Environment
from sdirl.inference_tasks import BolfiParams
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
    bolfi, store = wrapper.construct_BOLFI(n_surrogate_samples=20, batch_size=1, client=client)
    return bolfi, loc


class TestBOLFI_ML_ComparisonExperiment():

    def setup_method(self, method):
        self.env = Mock(Environment)
        rs = np.random.RandomState(0)
        self.env.setup = Mock(return_value=(rs, None))
        self.seed = 0
        self.cmdargs = list()
        self.model = SimpleGaussianModel(["mean"])
        self.ground_truth = [0]
        self.bolfi_params = BolfiParams(10, 1)
        self.exp = BOLFI_ML_ComparisonExperiment(self.env,
                self.seed,
                self.cmdargs,
                self.model,
                self.ground_truth,
                self.bolfi_params)

    def test_initialization_calls_environment_setup(self):
        assert self.env.setup.called

    @slow  # ~2min
    def test_running_generates_reasonable_results(self):
        self.exp.run()
        assert np.abs(self.exp.results["posteriors_disc"][-1].ML[0] - self.ground_truth[0]) < 0.1
        assert np.abs(self.exp.results["posteriors_logl"][-1].ML[0] - self.ground_truth[0]) < 0.1
        assert self.exp.results["errors_disc"][-1] < 0.1
        assert self.exp.results["errors_logl"][-1] < 0.1
        assert self.exp.results["duration_disc"] > 1e-3
        assert self.exp.results["duration_logl"] > 1e-3


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

