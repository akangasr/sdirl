
import numpy as np
import random

from sdirl.inference_tasks import BOLFIModelWrapper
from sdirl.model import SimpleGaussianModel

import logging

class TestBOLFIModelWrapper():

    def performs_sensible_optimization_with_simple_model(self, approximate):
        np.random.seed(1)
        elfi_logger = logging.getLogger("elfi.methods")
        elfi_logger.setLevel(logging.DEBUG)
        model = SimpleGaussianModel(["mean"])
        rs = np.random.RandomState(0)
        loc = [0.0]
        obs = model.simulate_observations(loc, rs)
        wrapper = BOLFIModelWrapper(model, obs, approximate=approximate)
        bolfi = wrapper.construct_BOLFI(n_surrogate_samples=10, batch_size=1, client=None)
        posterior = bolfi.infer()
        assert np.abs(posterior.ML[0] - loc[0]) < 0.1

    def test_performs_sensible_optimization_with_simple_model_using_likelihood(self):
        self.performs_sensible_optimization_with_simple_model(True)

    def test_performs_sensible_optimization_with_simple_model_using_discrepancy(self):
        self.performs_sensible_optimization_with_simple_model(False)
