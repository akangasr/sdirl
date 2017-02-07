import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
    )

import GPy

from elfi.bo.gpy_model import GPyModel
from sdirl.bolfi_utils import *


def create_simple_gpmodel():
    X = np.random.uniform(-1, 1, 100)
    Y = np.random.normal(X**2, 0.1)
    gpmodel = GPyModel(input_dim=1,
            bounds=((-1, 1),),
            kernel_class=GPy.kern.RBF,
            kernel_var=1.0,
            kernel_scale=1.0,
            noise_var=0.5,
            optimizer="scg",
            max_opt_iters=50)
    gpmodel.update(X[:, None], Y[:, None])
    return gpmodel


class TestSerializableBolfiPosterior():

    def create_simple_posterior(self):
        gpmodel = create_simple_gpmodel()
        return SerializableBolfiPosterior(model=gpmodel, threshold=0.0)

    def test_sensible_posterior_can_be_constructed_from_simple_model(self):
        np.random.seed(1)
        posterior = self.create_simple_posterior()
        assert abs(posterior.ML - 0.0) < 0.01

    @staticmethod
    def assert_simple_posteriors_equal_within_small_margin(post1, post2):
        for loc in [[-0.9], [0.0], [0.9]]:
            logpdf1 = post1.logpdf(loc)
            logpdf2 = post2.logpdf(loc)
            assert np.abs(logpdf1 - logpdf2) < 1e-2
        assert np.abs(post1.threshold - post2.threshold) < 1e-2
        np.testing.assert_array_almost_equal(post1.ML, post2.ML, decimal=2)
        np.testing.assert_array_almost_equal(post1.MAP, post2.MAP, decimal=2)

    def test_serializing_then_deserializing_does_not_change_posterior(self):
        np.random.seed(1)
        posterior = self.create_simple_posterior()
        dict_post = posterior.to_dict()
        posterior2 = SerializableBolfiPosterior.from_dict(dict_post)
        self.assert_simple_posteriors_equal_within_small_margin(posterior, posterior2)

    def test_loading_a_posterior_from_dict_is_consistent(self):
        np.random.seed(1)
        posterior = self.create_simple_posterior()
        dict_post = posterior.to_dict()
        posterior2 = SerializableBolfiPosterior.from_dict(dict_post)
        posterior3 = SerializableBolfiPosterior.from_dict(dict_post)
        self.assert_simple_posteriors_equal_within_small_margin(posterior2, posterior3)

