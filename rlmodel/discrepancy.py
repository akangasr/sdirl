import numpy as np

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Compute discrepancy between summaries.
"""

class Discrepancy():

    def __init__(self, used_discrepancy_features):
        """ Calculates discrepancy for menu simulations

        Parameters
        ----------
        used_discrepancy_features : dict
            features to use: {string: bool}
        """
        self.used_discrepancy_features = used_discrepancy_features

    def __call__(self, obs, sim, ret_distr=False):
        """ ELFI interface.
        Sequential function.

        Parameters
        ----------
        obs : tuple of single np.ndarray containing a dict
            Observation data features
        sim : tuple of single np.ndarray containing a dict
            Simulated data features
        ret_distr : bool
            If True, also returns the distribution data (eg for printing)

        Returns
        -------
        Non-negative discrepancy value (and distribution if enabled)
        """
        assert type(obs) is tuple, type(obs)
        assert type(sim) is tuple, type(sim)
        assert len(obs) == 1, len(obs)
        assert len(sim) == 1, len(sim)
        obs = obs[0]
        sim = sim[0]
        assert type(obs) is np.ndarray, type(obs)
        assert type(sim) is np.ndarray, type(sim)
        assert len(obs) == 1, len(obs)
        assert len(sim) == 1, len(sim)
        obs = obs[0]
        sim = sim[0]
        assert type(obs) is dict, type(obs)
        assert type(sim) is dict, type(sim)
        disc = list()
        if ret_distr == True:
            distr = dict()
        for k in obs.keys():
            obs_k = obs.get(k)
            sim_k = sim.get(k)
            feature_type = None
            if "task_completion_time" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 3000.0
                nbins = 8
            elif "location_of_gaze_to_target" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 7.0
                nbins = 8
            elif "proportion_of_gaze_to_target" in k:
                feature_type = "graph"
                minbin = 0.0
                maxbin = 7.0
                nbins = 8
            elif "fixation_duration" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 1000.0
                nbins = 10
            elif "saccade_duration" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 150.0
                nbins = 10
            elif "number_of_saccades" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 14.0
                nbins = 15
            elif "fixation_locations" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 7.0
                nbins = 8
            elif "length_of_skips" in k:
                feature_type = "histogram"
                minbin = 0.0
                maxbin = 7.0
                nbins = 8
            else:
                raise ValueError("Unknown feature: {}".format(k))

            if feature_type == "histogram":
                bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
                obs_k_lim = [f if f < maxbin else maxbin+1e-10 for f in obs_k]
                sim_k_lim = [f if f < maxbin else maxbin+1e-10 for f in sim_k]
                obs_hist, obs_edges = np.histogram(obs_k_lim, bins=bins)
                sim_hist, sim_edges = np.histogram(sim_k_lim, bins=bins)
                obs_hist_norm = obs_hist / sum(obs_hist)
                sim_hist_norm = sim_hist / sum(sim_hist)
                # core discrepancy function
                disc_i = ((np.mean(obs_hist) - np.mean(sim_hist)) ** 2 + np.abs(np.std(obs_hist) - np.std(sim_hist))) / 1000000

            elif feature_type == "graph":
                bins = np.hstack((np.linspace(minbin, maxbin, nbins), [maxbin+(maxbin-minbin)/float(nbins)]))
                obs_hist_bins, obs_edges = np.histogram(list(), bins=bins)
                sim_hist_bins, sim_edges = np.histogram(list(), bins=bins)
                obs_hist_vals = [0] * len(obs_hist_bins)
                sim_hist_vals = [0] * len(sim_hist_bins)
                obs_hist_counts = [0] * len(obs_hist_bins)
                sim_hist_counts = [0] * len(sim_hist_bins)
                # assume minbin == 0, increment == 1
                for f in obs_k:
                    obs_hist_vals[f[0]] += f[1]
                    obs_hist_counts[f[0]] += 1
                for f in sim_k:
                    sim_hist_vals[f[0]] += f[1]
                    sim_hist_counts[f[0]] += 1
                obs_hist = list()
                for i in range(len(obs_hist_vals)):
                    if obs_hist_counts[i] == 0:
                        obs_hist.append(0)
                    else:
                        obs_hist.append(obs_hist_vals[i] / float(obs_hist_counts[i]))
                sim_hist = list()
                for i in range(len(sim_hist_vals)):
                    if sim_hist_counts[i] == 0:
                        sim_hist.append(0)
                    else:
                        sim_hist.append(sim_hist_vals[i] / float(sim_hist_counts[i]))
                obs_hist_norm = None
                sim_hist_norm = None
                # core discrepancy function
                disc_i = 0.0

            else:
                raise ValueError("Unknown feature type: {}".format(feature_type))

            if k in self.used_discrepancy_features and self.used_discrepancy_features[k] == 1:
                used = True
                disc.append(disc_i)
            else:
                used = False
            if ret_distr == True:
                distr[k] = {
                        "feature_type": feature_type,
                        "obs": obs,
                        "sim": sim,
                        "obs_hist": obs_hist,
                        "sim_hist": sim_hist,
                        "obs_edges": obs_edges,
                        "sim_edges": sim_edges,
                        "obs_hist_norm": obs_hist_norm,
                        "sim_hist_norm": sim_hist_norm,
                        "disc_i": disc_i,
                        "used": used
                        }
        if len(disc) > 0:
            # core discrepancy function
            d = np.mean(disc)
        else:
            logger.critical("No features to compute discrepancy!")
            d = 0.0
        logger.debug("discrepancy = {}".format(d))
        if ret_distr == True:
            return d, distr
        return np.atleast_1d([d])

