import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
    )

from functools import partial
import numpy as np
import GPy

from elfi.bo.gpy_model import GPyModel
from elfi import InferenceTask
from sdirl.elfi_utils import *


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

def MA2(t1, t2, batch_size=1, random_state=None):
    n_obs = 100
    if random_state is None:
        random_state = np.random.RandomState()
    w = random_state.randn(batch_size, n_obs+2) # i.i.d. sequence ~ N(0,1)
    y = w[:,2:] + t1 * w[:,1:-1] + t2 * w[:,:-2]
    return y

def autocov(x, lag=1):
    mu = np.mean(x, axis=1, keepdims=True)
    C = np.mean(x[:,lag:] * x[:,:-lag], axis=1, keepdims=True) - mu**2.
    return C

def distance(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=2, axis=0)
    return d

def create_simple_inference_task():
    itask = InferenceTask()
    random_state = np.random.RandomState(20161130)
    y = MA2(1.0, 1.0, random_state=random_state)
    t1 = elfi.Prior('t1', 'uniform', 0, 2, inference_task=itask)
    t2 = elfi.Prior('t2', 'uniform', 0, 2, inference_task=itask)
    Y = elfi.Simulator('MA2', MA2, t1, t2, observed=y, inference_task=itask)
    s1 = elfi.Summary('s1', autocov, Y, inference_task=itask)
    autocov2 = partial(autocov, lag=2)
    s2 = elfi.Summary('s2', autocov2, Y, inference_task=itask)
    d = elfi.Discrepancy('d', distance, s1, s2, inference_task=itask)
    itask.parameters = [t1, t2]
    return itask


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



class TestBolfiFactory():

    def get_default_bolfi_params_for_simple_inference_task(self):
        params = BolfiParams()
        params.bounds = ((0, 2), (0, 2))
        params.n_surrogate_samples = 10
        params.batch_size = 1
        params.sync = True
        params.kernel_class = GPy.kern.RBF
        params.noise_var = 0.01
        params.kernel_var = 1.2
        params.kernel_scale = 1.0
        params.gp_params_optimizer = "scg"
        params.gp_params_max_opt_iters = 100
        params.exploration_rate = 1.5
        params.acq_opt_iterations = 200
        params.rbf_scale = 0.1
        params.rbf_amplitude = 0.5
        params.batches_of_init_samples = 1
        params.inference_type = InferenceType.ML
        params.client = None
        params.use_store = True
        return params

    def assert_bolfi_matches_params(self, bolfi, params):
        assert bolfi.model.bounds == params.bounds
        assert bolfi.acquisition.samples_left == params.n_surrogate_samples
        assert len(bolfi.acquisition.schedule) == 2
        assert bolfi.acquisition.schedule[0].samples_left == params.batch_size * params.batches_of_init_samples
        assert bolfi.acquisition.schedule[1].exploration_rate == params.exploration_rate
        assert isinstance(bolfi.acquisition.schedule[0], RandomAcquisition)
        if params.sync is True:
            if params.inference_type == InferenceType.ML:
                assert isinstance(bolfi.acquisition.schedule[1], BolfiMLAcquisition)
            else:
                assert isinstance(bolfi.acquisition.schedule[1], BolfiAcquisition)
        else:
            assert bolfi.acquisition.schedule[1].rbf_scale == params.rbf_scale
            assert bolfi.acquisition.schedule[1].rbf_amplitude == params.rbf_amplitude
            if params.inference_type == InferenceType.ML:
                assert isinstance(bolfi.acquisition.schedule[1], AsyncBolfiMLAcquisition)
            else:
                assert isinstance(bolfi.acquisition.schedule[1], AsyncBolfiAcquisition)
        assert bolfi.batch_size == params.batch_size
        assert bolfi.sync == params.sync
        if bolfi.model.gp is not None:
            assert bolfi.model.gp.kern.__class__ == params.kernel_class
            assert bolfi.model.gp.likelihood.variance == params.noise_var
            assert bolfi.model.gp.kern.variance == params.kernel_var
            assert bolfi.model.gp.kern.lengthscale == params.kernel_scale
        assert bolfi.model.optimizer == params.gp_params_optimizer
        assert bolfi.model.max_opt_iters == params.gp_params_max_opt_iters
        if params.client is not None:
            assert bolfi.client == params.client
        else:
            assert bolfi.client is not None
        if params.use_store is True:
            assert bolfi.store is not None
        else:
            assert bolfi.store is None

    @slow  # ~5s
    def test_constructs_correct_inference_for_simple_model(self):
        np.random.seed(1)
        itask = create_simple_inference_task()
        params = self.get_default_bolfi_params_for_simple_inference_task()
        bf = BolfiFactory(itask, params)
        bolfi = bf.get_new_instance()
        self.assert_bolfi_matches_params(bolfi, params)
        bolfi.infer()

    def test_raises_error_when_task_has_no_parameters(self):
        np.random.seed(1)
        itask = create_simple_inference_task()
        itask.parameters = list()
        params = self.get_default_bolfi_params_for_simple_inference_task()
        with pytest.raises(ValueError):
            bf = BolfiFactory(itask, params)

    def test_raises_error_when_task_parameters_do_not_match_bounds(self):
        np.random.seed(1)
        itask = create_simple_inference_task()
        params = self.get_default_bolfi_params_for_simple_inference_task()
        params.bounds = params.bounds + ((0,1),)  # add extra bound
        with pytest.raises(ValueError):
            bf = BolfiFactory(itask, params)

