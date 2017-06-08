import numpy as np
import GPy
from enum import IntEnum

import elfi
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.acquisition import *
from elfi.methods.methods import BOLFI
from elfi.methods.results import BolfiPosterior
from elfi.model.tools import vectorize
from elfi.model.elfi_model import ElfiModel, ComputationContext

from sdirl.environment import Environment

import logging
logger = logging.getLogger(__name__)

""" Extensions and helper functions for the ELFI library.

InferenceTaskFactory - Constructs an elfi.InferenceTask from sdirl.ModelBase derivative
SerializableBolfiPosterior - Extends elfi.BolfiPosterior to make it (de)serializable
BolfiParams - Encapsulates various elfi.BOLFI parameters
BolfiFactory - Constructs elfi.BOLFI using BolfiParams and an elfi.InferenceTask
"""

class GridAcquisition(AcquisitionBase):
    """Acquisition from a grid.

    Parameters
    ----------
    tics : list of lists containing axis locations defining the grid
           example: [[1,2,3], [2,4]] -> grid (1,2), (1,4), (2,2), (2,4), (3,2), (3,4)
    """

    def __init__(self, tics, *args, **kwargs):
        self.tics = tics
        self.next_id = 0
        n_priors = len(prior_list)

        # hacky...
        class DummyModel(object):
            pass
        model = DummyModel()
        model.input_dim = len(self.tics)
        model.bounds = [(l[0], l[-1]) for l in self.tics]
        model.evaluate = lambda x : exec('raise NotImplementedError')

        super(GridAcquisition, self).__init__(*args, model=model, **kwargs)

    def acquire(self, n_values, pending_locations=None):
        ret = super(RandomAcquisition, self).acquire(n_values, pending_locations)
        for i, p in enumerate(self.prior_list):
            idx = self.next_id
            for j, tics in enumerate(self.tics):
                l = len(tics)
                mod = l % idx
                idx /= l
                ret[j, i] = tics[mod]
        logger.debug("Acquired {}".format(n_values))
        return ret



class ElfiModelFactory():
    """ Factory for constructing ElfiModel objects from ModelBase

    Parameters
    ----------
    model : ModelBase
    """
    def __init__(self, model):
        self.model = model

    def get_new_instance(self):
        elfimodel = ElfiModel(name="model")

        parameters = list()
        inf_parameters = list()
        for parameter in self.model.parameters:
            if parameter.bounds[0] > parameter.bounds[1]:
                raise ValueError("Parameter bounds must be in correct order, received {}".format(parameter.bounds))
            elif parameter.bounds[0] == parameter.bounds[1]:
                param = elfi.Constant(name=parameter.name,
                                      value=parameter.bounds[0],
                                      model=elfimodel)
            else:
                param = elfi.Prior(parameter.prior.distribution_name,
                                   *parameter.prior.params,
                                   name=parameter.name,
                                   model=elfimodel)
                inf_parameters.append(param)
            parameters.append(param)
        logger.info("Parameters: {}".format(parameters))

        if self.model.observation is not None:
            y = self.model.observation
        elif self.model.ground_truth is not None:
            y = self.model.simulator(*self.model.ground_truth, random_state=random_state)
        else:
            raise ValueError("Model should have observation or ground truth")

        Y = elfi.Simulator(vectorize(self.model.simulator),
                           *parameters,
                           observed=y,
                           name="simulator",
                           model=elfimodel)
        summaries = list()
        for summary in self.model.summaries:
            summaries.append(elfi.Summary(vectorize(summary.function),
                                          Y,
                                          name=summary.name,
                                          model=elfimodel))
        d = elfi.Discrepancy(vectorize(self.model.discrepancy),
                             *summaries,
                             name="discrepancy",
                             model=elfimodel)
        elfimodel.parameters = [p.name for p in inf_parameters]
        return elfimodel


class SerializableBolfiPosterior(BolfiPosterior):  # TODO: add this to elfi?
    """ Extends BolfiPosterior so that it can be serialized, deserialized and constructed from a model
    """

    @staticmethod
    def from_store(store):
        """ Constructs SerializableBolfiPosteriors from bolfi store, returns as a list
        """
        ret = list()
        idx = 0
        while True:
            model = store.get("BOLFI-model", idx)[0]
            if model is None:
                return ret
            opt_iters = 10000
            if idx < 10:
                opt_iters = 100
            ret.append(SerializableBolfiPosterior.from_model(model, opt_iters))
            idx += 1

    @staticmethod
    def from_model(model, opt_iters=10000):
        """ Constructs a SerializableBolfiPosterior from a compatible gp model
        """
        return SerializableBolfiPosterior(model, None, max_opt_iters=opt_iters)

    def to_dict(self):
        return {"none": "none"}  # TODO
        # hacky
        data = {
            "X_params": self.model._gp.X.tolist() if self.model._gp is not None else [],
            "Y_disc": self.model._gp.Y.tolist() if self.model._gp is not None else [],
            "kernel_class": self.model.kernel_class.__name__,
            "kernel_var": self.model.kernel_var,
            "kernel_scale": self.model.kernel_scale,
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
        model = GPyRegression(input_dim=len(bounds),
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
            n_BO_samples=0,
            n_random_samples=0,
            n_grid_samples=0,
            grid_axis=None,
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
            inference_type=InferenceType.FULL_POSTERIOR,
            client=None,
            use_store=True):
        self.bounds = bounds
        self.n_BO_samples = n_BO_samples
        self.n_random_samples = n_random_samples
        self.n_grid_samples = n_grid_samples
        self.grid_axis = grid_axis
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
        self.inference_type = inference_type
        self.client = client
        self.use_store = use_store

    def to_dict(self):
        return {
            "bounds": self.bounds,
            "n_BO_samples": self.n_BO_samples,
            "n_random_samples": self.n_random_samples,
            "n_grid_samples": self.n_grid_samples,
            "grid_axis": self.grid_axis,
            "batch_size": self.batch_size,
            "sync": self.sync,
            "kernel_class": self.kernel_class.__name__,
            "noise_var": self.noise_var,
            "kernel_var": self.kernel_var,
            "kernel_scale": self.kernel_scale,
            "gp_params_optimizer": self.gp_params_optimizer,
            "gp_params_max_opt_iters": self.gp_params_max_opt_iters,
            "exploration_rate": self.exploration_rate,
            "acq_opt_iterations": self.acq_opt_iterations,
            "rbf_scale": self.rbf_scale,
            "rbf_amplitude": self.rbf_amplitude,
            "inference_type": self.inference_type,
            #"client": self.client,  # TODO: serialization of client?
            "use_store": self.use_store
            }


class BolfiFactory():  # TODO: add this to elfi?
    """ Constructs an elfi.BOLFI inference object from BolfiParams and InferenceTask

    Parameters
    ----------
    model : ElfiModel
    params : BolfiParams
    """
    def __init__(self, model, params):
        self.model = model
        self.params = params
        if len(self.model.parameters) < 1:
            raise ValueError("Task must have at least one parameter.")
        if len(self.params.bounds) != len(self.model.parameters):
            raise ValueError("Task must have as many parameters (was {}) as there are bounds in parameters (was {})."\
                    .format(len(self.model.parameters), len(self.params.bounds)))

    def _get_new_gpmodel(self):
        return GPyRegression(input_dim=len(self.params.bounds),
                        bounds=self.params.bounds,
                        kernel_class=self.params.kernel_class,
                        kernel_var=self.params.kernel_var,
                        kernel_scale=self.params.kernel_scale,
                        noise_var=self.params.noise_var,
                        optimizer=self.params.gp_params_optimizer,
                        max_opt_iters=self.params.gp_params_max_opt_iters)

    def _get_new_acquisition(self, gpmodel):
        if self.params.n_random_samples > 0:
            assert False # TODO
            #acq = RandomAcquisition(self.task.parameters)
            #return acq
        if self.params.n_grid_samples > 0:
            assert False # TODO
            #acq = GridAcquisition(self.task.parameters)
            #return acq
        if self.params.n_BO_samples > 0:
            return LCBSC(delta=0.001,
                         model=gpmodel)

        logger.critical("No acquisition samples set, aborting!")
        assert False

    def _get_new_store(self):
        if self.params.use_store is True:
            return elfi.storage.DictListStore()
        return None

    def get_new_instance(self):
        """ Returns new BOLFI inference object
        """
        gpmodel = self._get_new_gpmodel()
        acquisition = self._get_new_acquisition(gpmodel)
#        store = self._get_new_store()
        return BOLFI(model=self.model,
                     target="discrepancy",
                     target_model=gpmodel,
                     acquisition_method=acquisition,
                     acq_noise_cov=0.1, # TODO
                     bounds=self.params.bounds,
                     initial_evidence=1, # TODO
                     update_interval=5, # TODO
                     batch_size=1,
                     max_parallel_batches=self.params.batch_size,
                     seed=self.params.seed)
