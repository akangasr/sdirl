
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

import json
import GPy

from distributed import Client
from functools import partial

import logging
logger = logging.getLogger(__name__)

class ParameterPrior():
    """ Encapsulated parameter prior

    Parameters
    ----------
    distribution_name : string, matching scipy.stats distribution names
    params : tuple of parameters for distribution
    """
    def __init__(self,
                 distribution_name = "uniform",
                 params = (0, 1)):
        self.distribution_name = distribution_name
        self.params = params

    def to_dict(self):
        """ Returns a json-serializable dict
        """
        return {
                "distribution_name" : self.distribution_name,
                "params" : self.params
                }


class ModelParameter():
    """ Encapsulated model parameter

    Parameters
    ----------
    name : string
    prior : ParameterPrior
    bounds : tuple (min value, max value)
    """
    def __init__(self,
                 name = "parameter",
                 bounds = (0, 1),
                 prior = None):
        self.name = name
        self.bounds = bounds
        self.prior = prior
        if self.prior is None:
            self.prior = ParameterPrior("uniform", (bounds[0], bounds[1]-bounds[0]))

    def to_dict(self):
        """ Returns a json-serializable dict
        """
        return {
                "name" : self.name,
                "prior" : self.prior.to_dict(),
                "bounds" : self.bounds
                }


class ObservationSummary():
    """ Encapsulated summary function

    Parameters
    ----------
    name : string
    function : callable f(observation) -> summary
    """
    def __init__(self,
                 name = "summary",
                 function = None):
        self.name = name
        self.function = function

    def to_dict(self):
        """ Returns a json-serializable dict
        """
        return {
                "name" : self.name,
                "function" : self.function.__name__,
                }


class ObservationDataset():
    """ Encapsulated observation dataset

    Parameters
    ----------
    data : contained data
    parameter_values (optional) : what values were used to simulate the dataset
    name (optional) : name for dataset
    """
    def __init__(self,
                 data = None,
                 parameter_values = None,
                 name = None):
        self.data = data
        self.parameter_values = parameter_values
        self.name = name

    def to_dict(self):
        """ Returns a json-serializable dict
        """
        if isinstance(self.data, list):
            ser_fun = getattr(self.data[0], "to_dict", None)
            if ser_fun is not None:
                serdata = [d.to_dict() for d in self.data]
            else:
                serdata = [str(d) for d in self.data]
        else:
            serdata = str(self.data)
        return {
                "data" : serdata,
                "name" : "" if self.name is None else self.name,
                "parameter_values" : "" if self.parameter_values is None else [str(v) for v in self.parameter_values],
                }


class ModelBase():
    """ Base class for models

    Parameters
    ----------
    name : string
    parameters : list of ModelParameter
    summaries : list of ObservationSummary
    simulator : callable f(*args, random_state) -> ndarray([ObservationDataset])
    summaries : list of ObservationSummary
    discrepancy : callable f(summaries1, summaries2) -> value
    observation : data
    """
    def __init__(self,
                 name = "model",
                 parameters = list(),
                 simulator = None,
                 summaries = list(),
                 discrepancy = None,
                 observation = None,
                 ground_truth = None):
        self.name = name
        self.parameters = parameters
        for parameter in self.parameters:
            if isinstance(parameter, ModelParameter) is False:
                logger.warning("Parameter {} does not implement ModelParameter"
                        .format(parameter))
        self.simulator = simulator
        self.summaries = summaries
        for summary in self.summaries:
            if isinstance(summary, ObservationSummary) is False:
                logger.warning("Summary {} does not implement ObservationSummary"
                        .format(summary))
        self.discrepancy = discrepancy
        self.observation = observation
        self.ground_truth = ground_truth

    def plot_obs(self, obs):
        pass

    def to_dict(self):
        """ Returns a json-serializable dict
        """
        return {
                "name" : self.name,
                "parameters" : [p.to_dict() for p in self.parameters],
                "simulator" : self.simulator.__name__,
                "summaries" : [s.to_dict() for s in self.summaries],
                "discrepancy" : self.discrepancy.__name__,
                "observation" : "" if self.observation is None else self.observation.to_dict(),
                "ground_truth" : "" if self.ground_truth is None else [str(v) for v in self.ground_truth],
                }


class DummyValue():
    """ Used to pass values from simulator to discrepancy function for
        computing the likelihood of the observations
    """
    def __init__(self, parameters, random_state):
        self.parameters = parameters
        self.random_state = random_state


class SDIRLModel(ModelBase):
    """ Interface for models used in SDIRL studies.

    Built using SDIRLModelFactory
    """

    def __init__(self):
        super(SDIRLModel, self).__init__()
        self.env = None
        self.task = None
        self.rl = None
        self.goal_state = None
        self.path_max_len = None
        self._paths = None
        self._prev_parameters = []
        self._precomp_obs_logprobs = dict()

    def to_dict(self):
        ret = super(SDIRLModel, self).to_dict()
        ret["env"] = self.env.to_dict()
        ret["task"] = self.task.to_dict()
        ret["rl"] = self.rl.to_dict()
        ret["goal_state"] = str(self.goal_state)
        ret["path_max_len"] = self.path_max_len
        return ret


    # Exact inference using logl
    def dummy_simulator(self, *parameters, random_state=None):
        return np.atleast_1d(DummyValue(parameters, random_state))

    def passthrough_summary_function(self, data):
        if isinstance(data[0], DummyValue):
            return data
        else:
            return self.summary_function(data)

    def logl_discrepancy(self, sim_data, obs_data):
        parameters = sim_data[0][0].parameters
        random_state = sim_data[0][0].random_state
        observations = obs_data[0][0].data
        return np.atleast_1d([-1 * self.evaluate_loglikelihood(parameters, observations, random_state)])

    def evaluate_loglikelihood(self, parameters, observations, random_state, scale=100.0):
        # Note: scaling != 1.0 will not preserve proportionality of likelihood
        # (only used as a hack to make GP implementation work, as it breaks with too large values)
        assert len(observations) > 0
        ind_log_obs_probs = list()
        policy = self._get_optimal_policy(parameters, random_state)
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
            paths = self.get_all_paths_for_obs(obs_i, policy)
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
        return sum(ind_log_obs_probs) / scale

    def _fill_path_tree(self, obs, full_path_len, policy, prune=True):
        """ Recursively fill path tree starting from obs
        """
        raise NotImplementedError("Subclass implements")

    def get_all_paths_for_obs(self, obs, policy=None):
        """ Returns a tree containing all possible paths that could have generated
            observation 'obs'.
        """
        self._paths = dict()
        start_time = time.time()
        self._fill_path_tree(obs, obs.path_len, policy)
        end_time = time.time()
        logger.info("Constructing path tree of depth {} took {} seconds"
                .format(obs.path_len, end_time-start_time))
        paths = self._paths
        self._paths = None
        return PathTreeIterator(obs, paths, obs.path_len)

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
        if len(path) < self.path_max_len:
            assert path.transitions[-1].next_state == self.goal_state  # should have been be pruned
        # assume all start states equally probable
        for transition in path.transitions:
            state = transition.prev_state
            action = transition.action
            next_state = transition.next_state
            act_i_prob = policy(state, action)
            tra_i_prob = self.env.transition_prob(transition)
            assert state != self.goal_state, (state, self.goal_state)  # should have been pruned
            assert act_i_prob != 0, (state, action)  # should have been pruned
            assert tra_i_prob != 0, (transition)  # should have been pruned
            logp += np.log(act_i_prob) + np.log(tra_i_prob)
        return np.exp(logp)

    def _get_optimal_policy(self, parameters, random_state):
        """ Returns a function pointer f(state, action) -> p(action|state) that defines the
            optimal policy.
        """
        self._train_model_if_needed(parameters, random_state)
        return self.rl.get_policy()


    # Approximate inference using discrepancy
    def simulate_observations(self, *parameters, random_state=None):
        self._train_model_if_needed(parameters, random_state)
        obs = self.rl.simulate(random_state=random_state)
        return np.atleast_1d([ObservationDataset(obs, parameters, "simulated")])

    def summary_function(self, data):
        raise NotImplementedError

    def calculate_discrepancy(self, sim_data, obs_data):
        raise NotImplementedError


    # Common functions
    def print_model(self):
        """ Do possible visualization of trained model
        """
        pass

    def _train_model_if_needed(self, parameters, random_state):
        """ Trains the model if parameters have changed
        """
        if not np.array_equal(self._prev_parameters, parameters):
            self.rl.train_model(parameters, random_state=random_state)
            self._prev_parameters = parameters
            self._paths = None
            self._precomp_obs_logprobs = dict()
            self.print_model()


class SDIRLModelFactory():
    """ Factory for constructing SDIRLModel objects
    """
    def __init__(self,
                 name,
                 parameters,
                 env,
                 task,
                 rl,
                 goal_state=None,
                 path_max_len=None,
                 klass=SDIRLModel,
                 observation=None,
                 ground_truth=None):
        self.name = name
        self.parameters = parameters
        self.env = env
        self.task = task
        self.rl = rl
        self.goal_state = goal_state
        self.path_max_len = path_max_len
        self.klass = klass
        self.observation = observation
        self.ground_truth = ground_truth
        if getattr(self.env, "to_dict", None) is None:
            raise ValueError("Env should implement to_dict()")
        if getattr(self.task, "to_dict", None) is None:
            raise ValueError("Task should implement to_dict()")
        if getattr(self.rl, "to_dict", None) is None:
            raise ValueError("RL should implement to_dict()")

    def get_new_instance(self, approximate):
        model = self.klass()
        model.name = self.name
        model.parameters = self.parameters
        model.observation = self.observation
        model.ground_truth = self.ground_truth
        model.env = self.env
        model.task = self.task
        model.rl = self.rl
        model.goal_state = self.goal_state
        model.path_max_len = self.path_max_len
        if approximate is True:
            model.simulator = model.simulate_observations
            model.summaries = [ObservationSummary("summary", model.summary_function)]
            model.discrepancy = model.calculate_discrepancy
        else:
            model.simulator = model.dummy_simulator
            model.summaries = [ObservationSummary("passthrough", model.passthrough_summary_function)]
            model.discrepancy = model.logl_discrepancy
        return model

