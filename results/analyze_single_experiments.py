#!/bin/python

import numpy as np
from enum import IntEnum
import os
import json

class GridSize(IntEnum):
    Grid3x3 = 1
    Grid5x5 = 2
    Grid7x7 = 3
    Grid9x9 = 4
    Grid11x11 = 5
    Grid13x13 = 6

class ExperimentLog():
    def __init__(self, grid_size, n_features, folder):
        self.grid_size = grid_size
        self.n_features = n_features
        self.folder = folder
        self.read_log()

    def read_log(self):
        with open(os.path.join(self.folder, "experiment.json")) as f:
            self.log = json.load(f)

    @property
    def duration(self):
        return self.log["results"]["results"]["duration"]

    @property
    def ml_error(self):
        return self.log["results"]["results"]["errors"][-1]

    def __str__(self):
        return "Experiment with {} and {} features at {}"\
            .format(self.grid_size.name, self.n_features, self.folder)

class ExperimentGroup():
    def __init__(self, experiments):
        self.exp = experiments
        self._get_grid_sizes()

    def print_experiments(self):
        print("Experiments:")
        for e in self.exp:
            print("* {}".format(e))

    def _get_grid_sizes(self):
        self.grid_sizes = list()
        for e in self.exp:
            if e.grid_size not in self.grid_sizes:
                self.grid_sizes.append(e.grid_size)
        self.grid_sizes = sorted(self.grid_sizes)

    def get_experiments_with_grid_size(self, grid_size):
        return [e for e in self.exp if e.grid_size == grid_size]

    @staticmethod
    def report_property_mean_std(exp, prop):
        values = [getattr(e, prop) for e in exp]
        return "mean {}, std {}".format(np.mean(values), np.std(values))

    def print_property_per_grid_size(self, prop):
        print("Statistics for {}".format(prop))
        for grid_size in self.grid_sizes:
            exp = self.get_experiments_with_grid_size(grid_size)
            rep = self.report_property_mean_std(exp, prop)
            print("* {} {}".format(grid_size.name, rep))


eg = ExperimentGroup([
        ExperimentLog(GridSize.Grid3x3, 3, "gd_3x3_3f_1"),
        ExperimentLog(GridSize.Grid3x3, 3, "gd_3x3_3f_2"),
        ExperimentLog(GridSize.Grid3x3, 3, "gd_3x3_3f_3"),
        ExperimentLog(GridSize.Grid5x5, 3, "gd_5x5_3f_1"),
        ExperimentLog(GridSize.Grid5x5, 3, "gd_5x5_3f_2"),
        ExperimentLog(GridSize.Grid5x5, 3, "gd_5x5_3f_3"),
        ])

eg.print_experiments()
eg.print_property_per_grid_size("duration")
eg.print_property_per_grid_size("ml_error")

