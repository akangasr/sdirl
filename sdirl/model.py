
import numpy as np
import scipy as sp
import sys
import os

import matplotlib.pyplot as plt

import dask
import GPy
import elfi
from elfi.wrapper import Wrapper
from elfi.bo.gpy_model import GPyModel

import json
import GPy

from distributed import Client
from functools import partial

import logging
logger = logging.getLogger(__name__)


class Model():
    """ Interface for models used in SDIRL studies

    Parameters
    ----------
    variable_names : list of strings
    """
    def __init__(self, variable_names):
        self.variable_names = variable_names
        self.n_var = len(self.variable_names)

    def evaluate_loglikelihood(self, variables, observations, random_state=None):
        """ Evaluates logarithm of unnormalized likelihood of variables given observations.

        Parameters
        ----------
        variables : list of values, matching variable_names
        observations : dataset compatible with model
        random_state : random value source

        Returns
        -------
        Log unnormalized likelihood
        """
        raise NotImplementedError("Subclass implements")

    def evaluate_discrepancy(self, variables, observations, random_state=None):
        """ Computes a discrepancy value that correlates with the
            likelihood of variables given observations.

        Parameters
        ----------
        variables : list of values, matching variable_names
        observations : dataset compatible with model
        random_state : random value source

        Returns
        -------
        Non-negative discrepancy value
        """
        sim_obs = self.simulate_observations(variables, random_state)
        return self.calculate_discrepancy(observations, sim_obs)

    def simulate_observations(self, variables, random_state):
        """ Simulates observations from model with variable values.

        Parameters
        ----------
        variables : list of values, matching variable_names
        random_state : random value source

        Returns
        -------
        Dataset compatible with model
        """
        raise NotImplementedError("Subclass implements")

    def calculate_discrepancy(self, observations, sim_observations):
        """ Evaluate discrepancy of two observations.

        Parameters
        ----------
        observations: dataset compatible with model
        sim_observations: dataset compatible with model

        Returns
        -------
        Non-negative discrepancy value
        """
        raise NotImplementedError("Subclass implements")

    def get_bounds(self):
        """ Return box constraints for the variables of the model.

        Returns
        -------
        n-tuple of 2-tuples of values, n = number of variables
        eg: ((0,1), (0,2), (1,2)) for 3 variables
        """
        raise NotImplementedError("Subclass implements")

    def get_elfi_gpmodel(self, approximate):
        """ Returns a gaussian process model for use in ELFI.

        Parameters
        ----------
        approximate : bool
            if true, returns model for discrepancy values
            if false, returns model for likelihood values
        """
        raise NotImplementedError("Subclass implements")

    def get_elfi_variables(self, inference_task):
        """ Returns a list of elfi.Prior objects for use in ELFI.

        Parameters
        ----------
        inference_task : elfi.InferenceTask
            inference task to add priors to
        """
        raise NotImplementedError("Subclass implements")



class RLModel(Model):
    """ RL based model
    """
    def evaluate_loglikelihood(variables, observations, random_state=None):
        ind_log_obs_probs = list()
        policy = self._get_optimal_policy(variables, random_state)
        for obs_i in observations:
            prob_i = 0.0
            paths = self._get_all_paths_for_obs(obs_i)
            for path in paths:
                prob_i += self._prob_obs(obs_i, path) * self._prob_path(path, policy)
            assert 0.0 <= prob_i <= 1.0, prob_i
            ind_log_obs_probs.append(np.log(prob_i))
        return sum(ind_log_obs_probs)

    def _get_optimal_policy(self, variables, random_state):
        """ Returns a function pointer f(state, action) -> p(action|state) that defines the
            optimal policy.
        """
        raise NotImplementedError("Subclass implements")

    def _get_all_paths_for_obs(self, observation):
        """ Returns an iterable containing all possible paths that could have generated
            'observation'.
        """
        raise NotImplementedError("Subclass implements")

    def _prob_obs(self, observation, path):
        """ Returns the probability that 'path' would generate 'observation'.
        """
        raise NotImplementedError("Subclass implements")

    def _prob_path(self, path, policy):
        """ Returns the probability that 'path' would have been generated given 'policy'.
        """
        raise NotImplementedError("Subclass implements")



class SimpleGaussianModel(Model):
    """ Simple example model, used for testing etc.
    """

    def __init__(self, variable_names):
        super(SimpleGaussianModel, self).__init__(variable_names)
        assert len(variable_names) == 1
        self.elfi_variables = list()
        self.scale = 0.1
        self.nval = 20

    def evaluate_loglikelihood(self, variables, observations, random_state=None):
        assert len(variables) == 1, variables
        ind_log_obs_probs = list()
        for obs_i in observations:
            prob_i = sp.stats.norm.pdf(obs_i, loc=variables[0], scale=self.scale)
            ind_log_obs_probs.append(np.log(prob_i))
        ret = sum(ind_log_obs_probs) / 1000.0
        return ret

    def simulate_observations(self, variables, random_state):
        assert len(variables) == 1, variables
        return random_state.normal(variables[0], scale=self.scale, size=self.nval)

    def calculate_discrepancy(self, observations1, observations2):
        return np.abs(np.mean(observations1) - np.mean(observations2))

    def get_bounds(self):
        return ((-2, 2),)

    def get_elfi_gpmodel(self, approximate):
        kernel_class = GPy.kern.RBF
        noise_var = 0.05
        model = GPyModel(input_dim=1,
                        bounds=self.get_bounds(),
                        kernel_class=kernel_class,
                        kernel_var=0.1,
                        kernel_scale=1.0,
                        noise_var=noise_var,
                        optimizer="scg",
                        max_opt_iters=0)
        return model

    def get_elfi_variables(self, inference_task):
        if len(self.elfi_variables) > 0:
            return self.elfi_variables
        bounds = self.get_bounds()
        for v, b in zip(self.variable_names, bounds):
            v = elfi.Prior(v, "uniform", b[0], b[1]-b[0], inference_task=inference_task)
            self.elfi_variables.append(v)
        return self.elfi_variables

