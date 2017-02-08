import os
import sys
import numpy as np
import random
import json
import GPy
import time
from enum import IntEnum

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.bo.acquisition import *
from elfi.methods import BOLFI, BolfiAcquisition, AsyncBolfiAcquisition
from elfi.posteriors import BolfiPosterior

import dask
from distributed import Client

import logging
logger = logging.getLogger(__name__)

""" Extensions and helper functions for the ELFI library
"""

class SerializableBolfiPosterior(BolfiPosterior):  # TODO: add this to elfi?
    """ Extends BolfiPosterior so that it can be serialized, deserialized and constructed from a model
    """

    @staticmethod
    def from_model(model):
        """ Constructs a SerializableBolfiPosterior from a compatible gp model
        """
        return SerializableBolfiPosterior(model, None)

    def to_dict(self):
        # hacky
        data = {
            "X_params": self.model.gp.X.tolist() if self.model.gp is not None else [],
            "Y_disc": self.model.gp.Y.tolist() if self.model.gp is not None else [],
            "kernel_class": self.model.kernel.__class__.__name__,
            "kernel_var": float(self.model.kernel.variance),
            "kernel_scale": float(self.model.kernel.lengthscale),
            "noise_var": self.model.noise_var,
            "threshold": self.threshold,
            "bounds": self.model.bounds,
            "ML": self.ML.tolist(),
            "ML_val": float(self.ML_val),
            "MAP": self.MAP.tolist(),
            "MAP_val": float(self.MAP_val),
            "optimizer": self.model.optimizer,
            "max_opt_iters": self.model.max_opt_iters
            }
        return data

    @staticmethod
    def from_dict(data):
        """ Constructs a SerializableBolfiPosterior from a dictionary
        """
        # hacky
        bounds = data["bounds"]
        kernel_class = eval("GPy.kern.{}".format(data["kernel_class"]))
        kernel_var = data["kernel_var"]
        kernel_scale = data["kernel_scale"]
        noise_var = data["noise_var"]
        optimizer = data["optimizer"]
        max_opt_iters = data["max_opt_iters"]
        model = GPyModel(input_dim=len(bounds),
                        bounds=bounds,
                        kernel_class=kernel_class,
                        kernel_var=kernel_var,
                        kernel_scale=kernel_scale,
                        noise_var=noise_var,
                        optimizer=optimizer,
                        max_opt_iters=max_opt_iters)
        X = np.atleast_2d(data["X_params"])
        Y = np.atleast_2d(data["Y_disc"])
        model.update(X, Y)
        posterior = BolfiPosterior(model, data["threshold"])
        posterior.ML = np.atleast_1d(data["ML"])
        posterior.ML_val = data["ML_val"]
        posterior.MAP = np.atleast_1d(data["MAP"])
        posterior.MAP_val = data["MAP_val"]
        return posterior


class InferenceType(IntEnum):
    ML = 1
    MAP = 2
    FULL_POSTERIOR = 3


class BolfiParams():  # TODO: add this to elfi?
    """ Encapsulates BOLFI parameters
    """
    def __init__(self,
            bounds=((0,1),),
            n_surrogate_samples=1,
            batch_size=1,
            sync=True,
            kernel_class=GPy.kern.RBF,
            noise_var=0.05,
            kernel_var=0.05,
            kernel_scale=0.1,
            gp_params_optimizer="scg",
            gp_params_max_opt_iters=50,
            exploration_rate=2.0,
            acq_opt_iterations=100,
            rbf_scale=1.0,
            rbf_amplitude=1.0,
            batches_of_init_samples=1,
            inference_type=InferenceType.FULL_POSTERIOR,
            client=None,
            use_store=True):
        self.bounds = bounds
        self.n_surrogate_samples = n_surrogate_samples
        self.batch_size = batch_size
        self.sync = sync
        self.kernel_class = kernel_class
        self.noise_var = noise_var
        self.kernel_var = kernel_var
        self.kernel_scale = kernel_scale
        self.gp_params_optimizer = gp_params_optimizer
        self.gp_params_max_opt_iters = gp_params_max_opt_iters
        self.exploration_rate = exploration_rate
        self.acq_opt_iterations = acq_opt_iterations
        self.rbf_scale = rbf_scale
        self.rbf_amplitude = rbf_amplitude
        self.batches_of_init_samples = batches_of_init_samples
        self.inference_type = inference_type
        self.client = client
        self.use_store = use_store

    @property
    def number_of_init_and_acq_samples(self):
        n_init_samples = min(self.batch_size * self.batches_of_init_samples,
                             self.n_surrogate_samples)
        n_acq_samples = self.n_surrogate_samples - n_init_samples
        return n_init_samples, n_acq_samples


class BolfiMLAcquisition(SecondDerivativeNoiseMixin, LCBAcquisition):
    """ Acquisition function for synchronous ML inference with BOLFI
    """
    pass


class AsyncBolfiMLAcquisition(RbfAtPendingPointsMixin, LCBAcquisition):
    """ Acquisition function for asynchronous ML inference with BOLFI
    """
    pass


class BolfiFactory():  # TODO: add this to elfi?
    """ Constructs an elfi.BOLFI inference object from BolfiParams and InferenceTask

    Parameters
    ----------
    task : elfi.InferenceTask
    params : BolfiParams
    """
    def __init__(self, task, params):
        self.task = task
        self.params = params
        if len(self.task.parameters) < 1:
            raise ValueError("Task must have at least one parameter.")
        if len(self.params.bounds) != len(self.task.parameters):
            raise ValueError("Task must have as many parameters (was {}) as there are bounds in parameters (was {})."\
                    .format(len(self.task.parameters), len(self.params.bounds)))

    def _get_new_gpmodel(self):
        return GPyModel(input_dim=len(self.params.bounds),
                        bounds=self.params.bounds,
                        kernel_class=self.params.kernel_class,
                        kernel_var=self.params.kernel_var,
                        kernel_scale=self.params.kernel_scale,
                        noise_var=self.params.noise_var,
                        optimizer=self.params.gp_params_optimizer,
                        max_opt_iters=self.params.gp_params_max_opt_iters)

    def _get_new_acquisition(self, gpmodel):
        n_init_samples, n_acq_samples = self.params.number_of_init_and_acq_samples
        acquisition_init = RandomAcquisition(self.task.parameters,
                                             n_samples=n_init_samples)
        acquisition_init.model = gpmodel  # TODO: fix elfi acquisition to handle this correctly
        if n_acq_samples < 1:
            logger.warning("Only initial random samples")
            return acquisition_init
        if self.params.sync is True:
            if self.params.inference_type == InferenceType.ML:
                cls = BolfiMLAcquisition
            else:
                cls = BolfiAcquisition
            acquisition = cls(
                    exploration_rate=self.params.exploration_rate,
                    opt_iterations=self.params.acq_opt_iterations,
                    model=gpmodel,
                    n_samples=n_acq_samples)
        else:
            if self.params.inference_type == InferenceType.ML:
                cls = AsyncBolfiMLAcquisition
            else:
                cls = AsyncBolfiAcquisition
            acquisition = cls(
                    exploration_rate=self.params.exploration_rate,
                    opt_iterations=self.params.acq_opt_iterations,
                    rbf_scale=self.params.rbf_scale,
                    rbf_amplitude=self.params.rbf_amplitude,
                    model=gpmodel,
                    n_samples=n_acq_samples)
        return acquisition_init + acquisition

    def _get_new_store(self):
        if self.params.use_store is True:
            return elfi.storage.DictListStore()
        return None

    def get_new_instance(self):
        """ Returns new BOLFI inference object
        """
        gpmodel = self._get_new_gpmodel()
        acquisition = self._get_new_acquisition(gpmodel)
        store = self._get_new_store()
        return BOLFI(distance_node=self.task.discrepancy,
                     parameter_nodes=self.task.parameters,
                     batch_size=self.params.batch_size,
                     store=store,
                     model=gpmodel,
                     acquisition=acquisition,
                     sync=self.params.sync,
                     bounds=self.params.bounds,
                     client=self.params.client)

