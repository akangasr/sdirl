
import numpy as np
import scipy as sp
import sys
import os
import time

import matplotlib.pyplot as plt

import dask
import GPy
import elfi
from elfi.wrapper import Wrapper
from elfi.bo.gpy_model import GPyModel

from sdirl.rl.utils import PathTreeIterator
from sdirl.gridworldmodel.mdp import State

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

    def to_dict(self):
        """ Returns a json-serialized dict
        """
        ret = {
                "variable_names" : self.variable_names,
                "n_var" : self.n_var,
                }
        return ret

    def evaluate_loglikelihood(self, variables, observations, random_state):
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

    def evaluate_discrepancy(self, variables, observations, random_state):
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


class ELFIModel(Model):

    def __init__(self, variable_names):
        super(ELFIModel, self).__init__(variable_names)
        self.kernel_class = GPy.kern.RBF
        self.noise_var = 0.05
        self.kernel_var = 0.05
        self.kernel_scale = 0.1
        self.optimizer = "scg"
        self.max_opt_iters = 50

    def to_dict(self):
        ret = super(ELFIModel, self).to_dict()
        ret["kernel_class"] = self.kernel_class.__name__
        ret["noise_var"] = self.noise_var
        ret["kernel_var"] = self.kernel_var
        ret["kernel_scale"] = self.kernel_scale
        ret["optimizer"] = self.optimizer
        ret["max_opt_iters"] = self.max_opt_iters
        return ret

    def get_elfi_gpmodel(self, approximate):
        """ Returns a gaussian process model for use in ELFI.

        Parameters
        ----------
        approximate : bool
            if true, returns model for discrepancy values
            if false, returns model for likelihood values
        """
        return GPyModel(input_dim=len(self.variable_names),
                        bounds=self.get_bounds(),
                        kernel_class=self.kernel_class,
                        kernel_var=self.kernel_var,
                        kernel_scale=self.kernel_scale,
                        noise_var=self.noise_var,
                        optimizer=self.optimizer,
                        max_opt_iters=self.max_opt_iters)

    def get_elfi_variables(self, inference_task):
        """ Returns a list of elfi.Prior objects for use in ELFI.

        Parameters
        ----------
        inference_task : elfi.InferenceTask
            inference task to add priors to
        """
        elfi_variables = list()
        bounds = self.get_bounds()
        for v, b in zip(self.variable_names, bounds):
            start = b[0]
            width = b[1] - b[0]
            v = elfi.Prior(v, "uniform", start, width, inference_task=inference_task)
            elfi_variables.append(v)
        return elfi_variables


class RLModel(Model):
    """ RL based model
    """

    def __init__(self, variable_names, verbose):
        super(RLModel, self).__init__(variable_names)
        self.verbose = verbose
        self.env = None
        self.task = None
        self.rl = None  # RLModel
        self.goal_state = None
        self.path_max_len = None
        self._precomp_paths = dict()
        self.prev_variables = []
        self._precomp_obs_logprobs = dict()

    def to_dict(self):
        ret = super(RLModel, self).to_dict()
        ret["verbose"] = int(self.verbose)
        ret["env"] = self.env.to_dict()
        ret["task"] = self.task.to_dict()
        ret["rl"] = self.rl.to_dict()
        ret["goal_state"] = self.goal_state
        ret["path_max_len"] = self.path_max_len
        return ret

    def evaluate_loglikelihood(self, variables, observations, random_state):
        assert len(observations) > 0
        ind_log_obs_probs = list()
        policy = self._get_optimal_policy(variables, random_state)
        logger.info("Evaluating loglikelihood of {} observations".format(len(observations)))
        start_time1 = time.time()
        for obs_i in observations:
            if obs_i in self._precomp_obs_logprobs.keys():
                logger.info("Using precomputed loglikelihood of {}".format(obs_i))
                logprob = self._precomp_obs_logprobs[obs_i]
                ind_log_obs_probs.append(logprob)
                continue
            logger.info("Evaluating loglikelihood of {}".format(obs_i))
            start_time2 = time.time()
            n_paths = 0
            prob_i = 0.0
            paths = self.get_all_paths_for_obs(obs_i)
            for path in paths:
                n_paths += 1
                prob_obs = self._prob_obs(obs_i, path)
                assert prob_obs > 0, "Paths should all have positive observation probability, but p({})={}"\
                        .format(path, prob_obs)
                prob_path = self._prob_path(path, policy)
                if prob_path > 0:
                    prob_i += prob_obs * prob_path
            assert 0.0 - 1e-10 < prob_i < 1.0 + 1e-10 , "Probability should be between 0 and 1 but was {}"\
                    .format(prob_i)
            logprob = np.log(prob_i)
            self._precomp_obs_logprobs[obs_i] = logprob
            ind_log_obs_probs.append(logprob)
            end_time2 = time.time()
            duration2 = end_time2 - start_time2
            logger.info("Processed {} paths in {} seconds ({} s/path)"
                    .format(n_paths, duration2, duration2/n_paths))
        end_time1 = time.time()
        duration1 = end_time1 - start_time1
        logger.info("Logl evaluated in {} seconds".format(duration1))
        return sum(ind_log_obs_probs)

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
        self._train_model_if_needed(variables, random_state)
        obs = self.rl.simulate(random_state=random_state)
        return self.summarize(obs)

    def summarize(self, raw_observation):
        """ Summary observation of simulated data
        """
        raise NotImplementedError("Subclass implements")

    def _get_optimal_policy(self, variables, random_state):
        """ Returns a function pointer f(state, action) -> p(action|state) that defines the
            optimal policy.
        """
        self._train_model_if_needed(variables, random_state)
        return self.rl.get_policy()

    def print_model(self):
        """ Do possible visualization of trained model
        """
        pass

    def _train_model_if_needed(self, variables, random_state):
        """ Trains the model if variables have changed
        """
        if not np.array_equal(self.prev_variables, variables):
            self.rl.train_model(*variables, random_state=random_state)
            self.prev_variables = variables
            self._precomp_obs_logprobs = dict()
            if self.verbose is True:
                self.print_model()

    def _fill_path_tree(self, obs):
        """ Recursively fill path tree starting from obs
        """
        raise NotImplementedError("Subclass implements")

    def get_all_paths_for_obs(self, obs):
        """ Returns a tree containing all possible paths that could have generated
            observation 'obs'.
        """
        self._fill_path_tree(obs)
        return PathTreeIterator(obs, self._precomp_paths, obs.path_len)

    def _prob_obs(self, observation, path):
        """ Returns the probability that 'path' would generate 'observation'.
        """
        raise NotImplementedError("Subclass implements")

    def _prob_path(self, path, policy):
        """ Returns the probability that 'path' would have been generated given 'policy'.

        Parameters
        ----------
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        policy : callable(state, action) -> p(action | state)
        """
        logp = 0
        if len(path) < self.path_max_len and path.transitions[-1].next_state != self.target_state:
            return 0.0
        # assume all start states equally probable
        for transition in path.transitions:
            state = transition.prev_state
            action = transition.action
            next_state = transition.next_state
            if state == self.target_state:
                # goal state can only be a 'next state' as it is absorbing
                return 0.0
            act_i_prob = policy(state, action)
            if act_i_prob == 0:
                return 0.0
            tra_i_prob = self.env.transition_prob(transition)
            if tra_i_prob == 0:
                return 0.0
            logp += np.log(act_i_prob) + np.log(tra_i_prob)
        return np.exp(logp)


class SimpleGaussianModel(ELFIModel):
    """ Simple example model, used for testing etc.
    """

    def __init__(self, variable_names):
        super(SimpleGaussianModel, self).__init__(variable_names)
        assert len(variable_names) == 1
        self.elfi_variables = list()
        self.scale = 0.1
        self.nval = 20
        self.kernel_scale = 1.0
        self.kernel_var = self.scale

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

