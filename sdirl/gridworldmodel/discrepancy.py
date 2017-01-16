import numpy as np
import scipy as sp
from collections import defaultdict
import json
from matplotlib import pyplot as pl
import random

import logging
logger = logging.getLogger("experiment")

from abc4py.discrepancy.discrepancy import Discrepancy
from abc4py.reporting.reporting_utils import *
from abc4py.utils.utils import *

from .feature_extraction import *

class Grid_discrepancy(Discrepancy):

    used_discrepancy_features = dict()

    def calculate(self, d1, d2, ret_distr=False):
        data1raw = d1.get()
        data2raw = d2.get()
        t1 = type(data1raw)
        t2 = type(data2raw)
        if t1 is not list and t2 is not list:
            data1 = get_feature_set(data1raw)
            data2 = get_feature_set(data2raw)
            return [self._calculate(data1, data2, ret_distr)]
        elif t1 is not list and t2 is list:
            r = list()
            data1 = get_feature_set(data1raw)
            for d in data2raw:
                data2 = get_feature_set(d)
                r.append(self._calculate(data1, data2, ret_distr))
            return r
        elif t1 is list and t2 is not list:
            r = list()
            data2 = get_feature_set(data2raw)
            for d in data1raw:
                data1 = get_feature_set(d)
                r.append(self._calculate(data1, data2, ret_distr))
            return r
        else:
            raise NotImplementedError("Discrepancy not implemented for types: %s %s" % (t1, t2))

    def _calculate(self, data1, data2, ret_distr=False):
        """ ABC discrepancy function """
        vals = list()
        if ret_distr == True:
            distr = dict()
        for k in data1.keys():
            f1 = data1.get(k)
            f2 = data2.get(k)
            feature_type = None
            if "number_of_actions_per_start" in k:
                feature_type = "graph"
                minbin = 0.0
                maxbin = 80.0
                nbins = 81
            elif "total_reward" in k:
                feature_type = "histogram"
                minbin = -1000.0
                maxbin = 1000.0
                nbins = 100
            elif "rewards" in k:
                feature_type = "histogram"
                minbin = -100.0
                maxbin = 100.0
                nbins = 100
            elif "number_of_visits_per_cell" in k:
                feature_type = "graph"
                minbin = 0.0
                maxbin = 80.0
                nbins = 81
            else:
                raise ValueError("Unknown feature: %s" % (k))

            if feature_type == "histogram":
                bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
                f1out = [f if f < maxbin else maxbin+1e-10 for f in f1]
                f2out = [f if f < maxbin else maxbin+1e-10 for f in f2]
                h1, e1 = np.histogram(f1out, bins=bins)
                h2, e2 = np.histogram(f2out, bins=bins)
                h1norm = h1 / sum(h1)
                h2norm = h2 / sum(h2)
                #disc = np.linalg.norm(np.array(h1norm) - np.array(h2norm)) / len(h1norm)
                disc = (abs(np.mean(f1) - np.mean(f2)) / 10.0) ** 2
            elif feature_type == "graph":
                bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
                h1h, e1 = np.histogram(list(), bins=bins)
                h2h, e2 = np.histogram(list(), bins=bins)
                h1r = [0] * len(h1h)
                h2r = [0] * len(h2h)
                n1 = [0] * len(h1h)
                n2 = [0] * len(h2h)
                # assume minbin == 0, increment == 1
                for f in f1:
                    h1r[f[0]] += f[1]
                    n1[f[0]] += 1
                for f in f2:
                    h2r[f[0]] += f[1]
                    n2[f[0]] += 1
                h1 = list()
                for i in range(len(h1r)):
                    if n1[i] == 0:
                        h1.append(0)
                    else:
                        h1.append(h1r[i] / float(n1[i]))
                h2 = list()
                for i in range(len(h2r)):
                    if n2[i] == 0:
                        h2.append(0)
                    else:
                        h2.append(h2r[i] / float(n2[i]))
                h1norm = None
                h2norm = None
                disc = (np.linalg.norm(np.array(h1) - np.array(h2)) ** 2) / len(h1)
            else:
                raise ValueError("Unknown feature type: %s" % (feature_type))

            if k in self.used_discrepancy_features and self.used_discrepancy_features[k] == 1:
                used = True
                vals.append(disc)
            else:
                used = False
            if ret_distr == True:
                distr[k] = {
                        "feature_type": feature_type,
                        "f1": f1,
                        "f2": f2,
                        "h1": h1,
                        "h2": h2,
                        "e1": e1,
                        "e2": e2,
                        #"h1p": h1p,
                        #"h2p": h2p,
                        "h1norm": h1norm,
                        "h2norm": h2norm,
                        #"h1pnorm": h1pnorm,
                        #"h2pnorm": h2pnorm,
                        "disc": disc,
                        "used": used
                        }
        if len(vals) > 0:
            d = np.mean(vals)
        else:
            logger.critical("No features to compute discrepancy!")
            d = 0.0
        logger.debug("discrepancy = %.2f" % (d))
        if ret_distr == True:
            return d, distr
        return d


    def get_avg_hist(self, var, histname, res):
        ret = list()
        for r in res:
            distr = r[1]
            for varname, hist in distr.items():
                if varname == var:
                    ret.append(np.array(hist[histname]))
        if histname in ["feature_type", "used"]:
            return ret[0]
        if histname == "disc":
            return np.array(ret)
        if histname in ["f1", "f2"]:
            # direct ravel doesn't seem to work if sublists are not of equal length
            # lists are of unequal length because amount of raw observations may vary between realizations
            r = list()
            for l in ret:
                r.extend(l)
            return r
        return np.mean(ret, axis=0)


    def print_data(self, data, fig):
        res = self.calculate(data, data, ret_distr=True)
        subplotrows = len(res[0][1])+1
        subplotcols = 1
        plotidx = 1
        for varname in res[0][1].keys():
            pl.subplot(subplotrows, subplotcols, plotidx)
            plotidx += 1
            used = self.get_avg_hist(varname, "used", res)
            feature_type = self.get_avg_hist(varname, "feature_type", res)
            color = "g" if used == True else "r"
            if feature_type == "histogram":
                bars = self.get_avg_hist(varname, "h1norm", res)
            elif feature_type == "graph":
                bars = self.get_avg_hist(varname, "h1", res)
            else:
                raise ValueError("Unknown feature type: %s" % (feature_type))
            bins = self.get_avg_hist(varname, "e1", res)
            plot_histogram(bars, bins, color)
            vals = self.get_avg_hist(varname, "f1", res)
            pl.title("%s\n(m=%.2f std=%.2f)" % (varname, np.mean(vals), np.std(vals)))
        pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)


    def get_ytics(self, varname):
        if "task_completion_time" in varname:
            return 1.0, 0.5
        elif "proportion_of_gaze_to_target" in varname:
            return 1.0, 0.5
        elif "fixation_duration" in varname:
            return 1.0, 0.5
        elif "number_of_saccades" in varname:
            return 1.0, 0.5
        return None, None

    def print_discrepancy(self, d_obs, d_sim, fig, task=None):
        txt = list()
        if task != None:
            variables = json.loads(task.variables_json)
            txt.append("variables: ")
            for v in variables:
                txt.append("%.3f, " % (v))

        res = self.calculate(d_obs, d_sim, ret_distr=True)
        disc = [r[0] for r in res]
        if len(disc) > 1:
            txt.append("discrepancy: mean=%.3f std=%.3f\n" % (np.mean(disc), np.std(disc)))
        else:
            txt.append("discrepancy: %.3f\n" % (disc[0]))
        subplotrows = len(res[0][1])+1
        subplotcols = 2
        plotidx = 1
        for varname in sorted(res[0][1].keys()):
            pl.subplot(subplotrows, subplotcols, plotidx)
            plotidx += 1
            used = self.get_avg_hist(varname, "used", res)
            feature_type = self.get_avg_hist(varname, "feature_type", res)
            color = "#008000" if used == True else "#00cdff"
            if feature_type == "histogram":
                bars = self.get_avg_hist(varname, "h2norm", res)
            if feature_type == "graph":
                bars = self.get_avg_hist(varname, "h2", res)
            bins = self.get_avg_hist(varname, "e2", res)
            scalemax, dt = self.get_ytics(varname)
            plot_histogram(bars, bins, color, scalemax=scalemax, dt=dt)
            vals = self.get_avg_hist(varname, "f2", res)
            pl.title("%s\nm=%.2f std=%.2f" % (varname, np.mean(vals), np.std(vals)))
            pl.subplot(subplotrows, subplotcols, plotidx)
            plotidx += 1
            color = "#805900" if used == True else "#ff8c00"
            if feature_type == "histogram":
                bars = self.get_avg_hist(varname, "h1norm", res)
            if feature_type == "graph":
                bars = self.get_avg_hist(varname, "h1", res)
            bins = self.get_avg_hist(varname, "e1", res)
            plot_histogram(bars, bins, color, scalemax=scalemax, dt=dt)
            vals = self.get_avg_hist(varname, "f1", res)
            divs = self.get_avg_hist(varname, "disc", res)
            if len(divs) > 1:
                pl.title("d: m=%.3f std=%.3f\nm=%.2f std=%.2f" % (np.mean(divs), np.std(divs), np.mean(vals), np.std(vals)))
            else:
                pl.title("d: %.3f\nm=%.2f std=%.2f" % (divs[0], np.mean(vals), np.std(vals)))
        pl.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

        #n_ses = 5
        #txt.append("Sample of %d random simulated sessions:\n" % (n_ses))
        #rnd_sessions = random.sample(d_sim.get()["sessions"], n_ses)
        #for session in rnd_sessions:
        #    txt.append("session string") # TODO

        fig.text(0.02, 0.01, "".join(txt))

