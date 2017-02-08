import os
import sys
import numpy as np
import random
import json
import GPy
import time
from copy import deepcopy

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.bo.acquisition import *
from elfi.methods import BOLFI
from elfi.posteriors import BolfiPosterior

import dask
from distributed import Client

from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import logging
logger = logging.getLogger(__name__)

import warnings


class Environment():
    """ Execution environment setup
    """
    __instance = None  # singleton

    def __new__(cls, args=list()):
        if Environment.__instance is None:
            Environment.__instance = object.__new__(cls)
            Environment.logging_setup()
            Environment.disable_pybrain_warnings()
        Environment.__instance.args = args
        Environment.__instance.random_state = Environment.__instance.rng_setup()
        Environment.__instance.client = Environment.__instance.client_setup()
        return Environment.__instance

    @staticmethod
    def get_instance():
        if Environment.__instance is None:
            return Environment()
        return Environment.__instance

    def to_dict(self):
        return {
                "args": self.args
                }

    @staticmethod
    def disable_pybrain_warnings():
        """ Ignore warnings from output
        """
        warnings.simplefilter("ignore")

    @staticmethod
    def logging_setup():
        """ Set logging
        """
        logger.setLevel(logging.INFO)
        model_logger = logging.getLogger("sdirl")
        model_logger.setLevel(logging.INFO)
        model_logger.propagate = False
        elfi_logger = logging.getLogger("elfi")
        elfi_logger.setLevel(logging.INFO)
        elfi_logger.propagate = False
        elfi_methods_logger = logging.getLogger("elfi.methods")
        elfi_methods_logger.setLevel(logging.DEBUG)
        elfi_methods_logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.propagate = False

        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        model_logger.handlers = [ch]
        elfi_logger.handlers = [ch]
        elfi_methods_logger.handlers = [ch]
        logger.handlers = [ch]

    def rng_setup(self):
        """ Return a random value source
        """
        if len(self.args) > 1:
            seed = self.args[1]
        else:
            seed = 0
        logger.info("Seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(random.randint(0, 10e7))
        return np.random.RandomState(random.randint(0, 10e7))

    def client_setup(self):
        """ Set up and return a dask client or None
        """
        client = None
        if len(self.args) > 2:
            address = "127.0.0.1:{}".format(int(self.args[2]))
            logger.info("Dask client at " + address)
            client = Client(address)
            dask.set_options(get=client.get)
        else:
            logger.info("Default dask client (client=None)")
        return client



