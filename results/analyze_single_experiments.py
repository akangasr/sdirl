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
        self._get_n_features()

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

    def _get_n_features(self):
        self.n_features = list()
        for e in self.exp:
            if e.n_features not in self.n_features:
                self.n_features.append(e.n_features)
        self.n_features = sorted(self.n_features)

    def get_experiments_with_grid_size(self, grid_size):
        return [e for e in self.exp if e.grid_size == grid_size]

    def get_experiments_with_n_features(self, n_features):
        return [e for e in self.exp if e.n_features == n_features]

    def get_experiments_with_grid_size_and_n_features(self, grid_size, n_features):
        return [e for e in self.exp if e.grid_size == grid_size and e.n_features == n_features]

    @staticmethod
    def report_property_mean_std(exp, prop):
        values = [getattr(e, prop) for e in exp]
        return "mean {}, std {}".format(np.mean(values), np.std(values))

    def print_property_per_grid_size(self, prop):
        print("Statistics for {} per grid size".format(prop))
        for grid_size in self.grid_sizes:
            exp = self.get_experiments_with_grid_size(grid_size)
            rep = self.report_property_mean_std(exp, prop)
            print("* {} {}".format(grid_size.name, rep))

    def print_property_per_n_features(self, prop):
        print("Statistics for {} per number of features".format(prop))
        for n_features in self.n_features:
            exp = self.get_experiments_with_n_features(n_features)
            rep = self.report_property_mean_std(exp, prop)
            print("* {} features {}".format(n_features, rep))

    def print_property_per_grid_size_and_n_features(self, prop):
        print("Statistics for {} per grid size and number of features".format(prop))
        for grid_size in self.grid_sizes:
            for n_features in self.n_features:
                exp = self.get_experiments_with_grid_size_and_n_features(grid_size, n_features)
                rep = self.report_property_mean_std(exp, prop)
                print("* {} {} features {}".format(grid_size.name, n_features, rep))


ex = list()
for i in [1,2,3,4,5]:
    for j in [1,2,3]:
        ex.append(ExperimentLog(GridSize.Grid3x3, j, "gd2_3x3_{}f_{}".format(j,i)))
        ex.append(ExperimentLog(GridSize.Grid5x5, j, "gd2_5x5_{}f_{}".format(j,i)))
        #ex.append(ExperimentLog(GridSize.Grid7x7, j, "gd2_7x7_{}f_{}".format(j,i)))
        #ex.append(ExperimentLog(GridSize.Grid11x11, j, "gd2_11x11_{}f_{}".format(j,i)))
        #ex.append(ExperimentLog(GridSize.Grid13x13, j, "gd2_13x13_{}f_{}".format(j,i)))
    for j in [1,2]:
        ex.append(ExperimentLog(GridSize.Grid7x7, j, "gd2_7x7_{}f_{}".format(j,i)))

eg = ExperimentGroup(ex)
eg.print_experiments()
eg.print_property_per_grid_size("duration")
eg.print_property_per_grid_size("ml_error")
eg.print_property_per_n_features("duration")
eg.print_property_per_n_features("ml_error")
eg.print_property_per_grid_size_and_n_features("duration")
eg.print_property_per_grid_size_and_n_features("ml_error")

