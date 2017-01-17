import numpy as np
import os
import json

import logging
logger = logging.getLogger("experiment")

from abc4py.data.data import Data

class Grid_data(Data):

    def get(self):
        if self.data is not None:
            return self.data
        try:
            loc_dir = os.path.dirname(os.path.realpath(__file__)),
            data_target = "%s/sim.json" % (loc_dir)
            with open(data_target) as data_file:
                self.data = json.load(data_file)
            logger.info("Loaded sim data")
        except:
            logger.info("Failed to load sim data")
            self.data = None
        return self.data

