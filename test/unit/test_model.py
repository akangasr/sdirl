
import numpy as np

import pytest

from sdirl.model import Model
from sdirl.model import SimpleGaussianModel

class TestModel():

    def test_can_be_initialized(self):
        model = Model(["var1"])

class TestSimpleGaussianModel():

    def test_simulation_results_are_sensible(self):
        model = SimpleGaussianModel(["mean"])
        rs = np.random.RandomState(1)
        for loc in [[-1.0], [0.0], [1.0]]:
            obs = model.simulate_observations(loc, rs)
            assert np.abs(np.mean(obs) - loc) < 0.5

    def test_likelihood_results_are_sensible(self):
        model = SimpleGaussianModel(["mean"])
        rs = np.random.RandomState(1)
        obs = list()
        loc = [[-1.0], [0.0], [1.0]]
        for l in loc:
            obs.append(model.simulate_observations(l, rs))
        assert model.evaluate_loglikelihood(loc[0], obs[0], rs) > model.evaluate_loglikelihood(loc[0], obs[1], rs)
        assert model.evaluate_loglikelihood(loc[0], obs[0], rs) > model.evaluate_loglikelihood(loc[0], obs[2], rs)
        assert model.evaluate_loglikelihood(loc[1], obs[1], rs) > model.evaluate_loglikelihood(loc[1], obs[0], rs)
        assert model.evaluate_loglikelihood(loc[1], obs[1], rs) > model.evaluate_loglikelihood(loc[1], obs[2], rs)
        assert model.evaluate_loglikelihood(loc[2], obs[2], rs) > model.evaluate_loglikelihood(loc[2], obs[0], rs)
        assert model.evaluate_loglikelihood(loc[2], obs[2], rs) > model.evaluate_loglikelihood(loc[2], obs[1], rs)

    def test_discrepancy_results_are_sensible(self):
        model = SimpleGaussianModel(["mean"])
        rs = np.random.RandomState(1)
        obs = list()
        loc = [[-1.0], [0.0], [1.0]]
        for l in loc:
            obs.append(model.simulate_observations(l, rs))
        assert model.evaluate_discrepancy(loc[0], obs[0], rs) < model.evaluate_discrepancy(loc[0], obs[1], rs)
        assert model.evaluate_discrepancy(loc[0], obs[0], rs) < model.evaluate_discrepancy(loc[0], obs[2], rs)
        assert model.evaluate_discrepancy(loc[1], obs[1], rs) < model.evaluate_discrepancy(loc[1], obs[0], rs)
        assert model.evaluate_discrepancy(loc[1], obs[1], rs) < model.evaluate_discrepancy(loc[1], obs[2], rs)
        assert model.evaluate_discrepancy(loc[2], obs[2], rs) < model.evaluate_discrepancy(loc[2], obs[0], rs)
        assert model.evaluate_discrepancy(loc[2], obs[2], rs) < model.evaluate_discrepancy(loc[2], obs[1], rs)


