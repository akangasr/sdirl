import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
    )

import numpy as np
import random
import os

import elfi

from sdirl.inference_tasks import BOLFIModelWrapper
from sdirl.inference_tasks import store_bolfi_posterior, load_bolfi_posterior
from sdirl.model import SimpleGaussianModel

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
    bolfi = wrapper.construct_BOLFI(n_surrogate_samples=20, batch_size=1, client=client)
    return bolfi, loc


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


class TestStoreLoadBolfiPosterior():

    @slow
    def test_storing_then_loading_does_not_change_posterior(self):
        np.random.seed(1)
        bolfi, l = create_simple_model_and_bolfi_inference(True)
        posterior = bolfi.infer()
        filename = "bolfi_posterior_test_file.json"
        if os.path.isfile(filename):
            os.remove(filename)
        store_bolfi_posterior(posterior, filename)
        posterior2 = load_bolfi_posterior(filename)
        for loc in [[0.1], [0.5], [1.0]]:
            logpdf_orig = posterior.logpdf(loc)
            logpdf_new = posterior2.logpdf(loc)
            assert np.abs(logpdf_orig - logpdf_new) < 1e-2
        assert np.abs(posterior.threshold - posterior2.threshold) < 1e-5
        np.testing.assert_array_almost_equal(posterior.ML, posterior2.ML, decimal=3)
        np.testing.assert_array_almost_equal(posterior.MAP, posterior2.MAP, decimal=3)

