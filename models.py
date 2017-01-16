import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dask
import GPy
import elfi
from elfi.wrapper import Wrapper
from elfi.bo.gpy_model import GPyModel

import json
import numpy as np
import GPy

from menumodel.observation import BaillyData
from menumodel.discrepancy import Discrepancy
from menumodel.rl_model import SearchTask, SearchEnvironment
from menumodel.rl_base import RLModel
from menumodel.summary import feature_extraction

import elfi
from elfi import InferenceTask
from elfi.bo.gpy_model import GPyModel
from elfi.methods import BOLFI
from elfi.posteriors import BolfiPosterior


from distributed import Client
from functools import partial

from menumodel.elfi_inference import do_inference, inference_task
from menumodel.elfi_inference import load_posterior, store_posterior

import logging
logger = logging.getLogger(__name__)

import warnings

def disable_pybrain_warnings():
    warnings.simplefilter("ignore")

def logging_setup():
    logger.setLevel(logging.INFO)
    logging.getLogger("menumodel").setLevel(logging.INFO)
    logging.getLogger("elfi").setLevel(logging.DEBUG)
    logging.getLogger("elfi.bo").setLevel(logging.INFO)
    elfi_log = logging.getLogger("elfi")
    elfi_log.setLevel(logging.INFO)
    elfi_log.propagate = False
    log = logging.getLogger("grid_world")
    log.setLevel(logging.INFO)
    log.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    elfi_log.handlers = [ch]
    log.handlers = [ch]




""" Model definitions.
"""

class Model():
    """ Interface for models.

    Parameters
    ----------
    variable_names : list of strings
    """
    def __init__(self, variable_names):
        self.variable_names = variable_names

    def evaluateLikelihood(variables, observations, random_state=None):
        """ Evaluates likelihood of variables given observations.

        Parameters
        ----------
        variables : list of values, matching variable_names
        observations : dataset compatible with model
        random_state : random value source

        Returns
        -------
        Unnormalized likelihood
        """
        ind_obs_probs = list()
        policy = self._get_optimal_policy(variables)
        for obs_i in observations:
            prob_i = 0.0
            paths = self._get_all_paths_for_obs(obs_i)
            for path in paths:
                prob_i += self._prob_obs(obs_i, path) * self._prob_path(path, policy)
            assert 0.0 <= prob_i <= 1.0, prob_i
            ind_obs_probs.append(prob_i)
        return np.prod(ind_obs_probs)

    def _get_optimal_policy(self, variables):
        """ Returns a list of all possible paths that could have generated
            observation 'obs'.
        """
        raise NotImplementedError("Subclass implements")

    def _get_all_paths_for_obs(self, obs):
        """ Returns an object containing all possible paths that could have generated
            observation 'obs'.
        """
        raise NotImplementedError("Subclass implements")

    def _prob_obs(self, obs, path):
        """ Returns the probability that 'path' would generate 'obs'.
        """
        raise NotImplementedError("Subclass implements")

    def _prob_path(self, path, policy):
        """ Returns the probability that 'path' would have been generated given 'policy'.
        """
        raise NotImplementedError("Subclass implements")

    def evaluateDiscrepancy(variables, observations, random_state=None):
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
        sim_obs = self.simulateObservations(variables, random_state)
        return self.calculateDiscrepancy(observations, sim_obs)

    def simulateObservations(variables, random_state):
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

    def calculateDiscrepancy(observations1, observations2):
        """ Evaluate discrepancy of two observations.

        Parameters
        ----------
        observations: dataset compatible with model

        Returns
        -------
        Non-negative discrepancy value
        """
        raise NotImplementedError("Subclass implements")


class WrapperModel(Model):
    """ Model that is based on an external simulator.
    """

    command = "echo [[{0}], [{seed}]]"

    def _simpostproc(x):
        ret = np.atleast_2d(eval(x), dtype=float)
        logger.debug("return {}".format(ret))
        return ret

    def simulateObservations(variables, random_state):
        assert len(variables) == len(self.variable_names), len(variables)
        seed = random_state.randint(1e10)
        cmd = self.command.format(*variables, seed=seed)
        simulator = Wrapper(cmd, self._simpostproc)
        obs = simulator()
        return obs


class MountainCarModel(WrapperModel):
    """ Mountain car model.
    """
    def __init__(variable_names=["goal_location"])
        super(MountainCarModel, self).__init__(variable_names)
        self.goal_min = 0.0
        self.goal_max = 2.0
        self.n_train = 10000
        self.n_obs = 10
        self.command = "cpp_models/build/toy_models mountain_car {0:.2f} {n_train} {n_obs} {seed}"
                .format(n_train=self.n_train, n_obs=self.n_obs)

    def evaluateLikelihood(variables, observations):
        raise NotImplementedError("TODO")

    def _get_stats(obs):
        mean = np.mean(obs, keepdims=True)
        std = np.std(obs, keepdims=True)
        return mean, std

    def calculateDiscrepancy(observations1, observations2):
        mean_loc1, std_loc1 = self._get_stats(observations1[:,:,0])
        mean_vel1, std_vel1 = self._get_stats(observations1[:,:,1])
        mean_loc2, std_loc2 = self._get_stats(observations2[:,:,0])
        mean_vel2, std_vel2 = self._get_stats(observations2[:,:,1])
        d_loc_mean = abs(100*mean_loc1 - 100*mean_loc2) ** 2
        d_loc_std = abs(100*std_loc1 - 100*std_loc2)
        d_vel_mean = abs(vel_sim - vel_obs) ** 2
        d_vel_std = abs(vel_sim - vel_obs)
        disc = d_loc_mean + d_loc_std + d_vel_mean + d_vel_std
        return disc

    def old_elfi_model()
        thr = elfi.Prior("thr", "uniform", throttle_min, throttle_max)
        Y = elfi.Simulator("MC", elfi.tools.vectorize(simulator), thr, observed=y)
        S1 = elfi.Summary("S1", loc, Y)
        S2 = elfi.Summary("S2", vel, Y)
        d = elfi.Discrepancy("d", distance, S1, S2)

        bounds = ((throttle_min, throttle_max),)
        input_dim = 1
        model = GPyModel(input_dim, bounds, kernel_var=0.5, kernel_scale=0.5, noise_var=0.001)
        acq = elfi.BolfiAcquisition(model,
                               exploration_rate=2.5,
                               n_samples=n_sim)
        bolfi = elfi.BOLFI(d, [thr],
                      batch_size=n_batch,
                      model=model,
                      acquisition=acq,
                      client=client)

        result = bolfi.infer()
        log.info("MAP at {}".format(result.MAP))
        return result

    def old_do_inference_mountain_car(client, outdir):
        np.random.seed(1234)
        val_min = 0.7
        val_max = 2.0
        delta_inf = 0.1
        delta_plt = 0.001
        locations = np.arange(val_min+delta_inf,
                              val_max+1e-10,
                              delta_inf)
        n_locations = locations.shape[0]
        n_samples = 10
        n_dim = 1
        n_sim = 100
        n_batch = 2
        estimates = np.empty((n_locations, n_samples), dtype=object)
        for i, location in enumerate(locations):
            log.info("Location {}".format(location))
            for j in range(n_samples):
                log.info("Sample {}..".format(j+1))
                estimates[i][j] = mountain_car(client,
                                               val_min,
                                               val_max,
                                               location,
                                               np.random.randint(10000),
                                               n_sim,
                                               n_batch)
            plot_estimates(estimates[i], location, np.arange(val_min, val_max+1e-10, delta_plt) , outdir)
        return locations, estimates


    def old_plot_estimates(estimates, location, locrange, outdir):
        post = np.zeros((len(estimates), len(locrange)))
        for i, est in enumerate(estimates):
            for j, loc in enumerate(locrange):
                post[i, j] = est.pdf(np.array([loc]))
            # approximate normalization
            post[i] /= sum(post[i])
        postmax = np.amax(post)
        fig = plt.figure()
        plt.xlim((locrange[0], locrange[-1]))
        plt.ylim((0, postmax * 1.05))
        plt.xlabel("Value (true {:.2f})".format(location))
        plt.ylabel("Probability density")
        for p in post:
            plt.plot(locrange, p, "b-")
        avgp = np.mean(post, axis=0)
        plt.plot(locrange, avgp, "r-")
        plt.axvline(location, color="g")
        plt.savefig("{}/post_{:.2f}.png".format(outdir, location))
        fig.clear()

    def old_main():
        client = Client()
        dask.set_options(get=client.get)
        if len(sys.argv) != 3:
            print("mountain_car.py <results_dir> <experiment_id>")
        outdir = sys.argv[1]
        ex_id = sys.argv[2]
        resdir = outdir + "/" + ex_id
        os.makedirs(resdir)
        do_inference_mountain_car(client, resdir)



class GridWorldModel(WrapperModel):
    """ Grid world model
    """
    def __init__(self, variable_names, world_seed):
        super(GridWorldModel, self).__init__(variable_names)
        self.n_train = 500000
        self.n_obs = 1000
        self.size = 10
        self.prob_rnd_move = 0.1
        self.world_seed = world_seed
        self.n_init_locs = (self.size - 1) * 4
        self.output_type = 0  # summaries
        self.precomp_paths = dict()
        self.target_x = self.size / 2
        self.target_y = self.size / 2

        self.command = "cpp_models/build/toy_models grid_world {size} {prob_rnd_move} {n_train} {n_obs} {output_type} {seed} {world_seed} {n_weights}"
                        .format(size=size,
                                prob_rnd_move=prob_rnd_move,
                                n_train=n_train,
                                n_obs=n_obs,
                                output_type=output_type,
                                world_seed=world_seed,
                                n_weights=len(self.variable_names))
        for i in range(len(self.variable_names)):
            command += " {" + str(i) + "}"

    class Observation():
        def __init__(self, start_state, path_len):
            self.start_state = path_len
            self.path_len = path_len

        def __eq__(a, b):
            return a.start_state == b.start_state and a.path_len == b.path_len

        def __hash__(self):
            # assume max grid size == 100
            return (self.state, self.path_len).__hash__()

    class Path():
        def __init__(self, states, actions):
            self.states = states
            self.actions = actions

    class State():
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(a, b):
            return a.x == b.x and a.y == b.y

        def __hash__(self):
            # assume max grid size == 100
            return (self.x, self.y).__hash__()

    class Transition():
        def __init__(self, state, action):
            self.state = state
            self.action = action

        def __eq__(a, b):
            return a.state == b.state and a.action == b.action

        def __hash__(self):
            # assume max grid size == 100
            return (self.state.x, self.state.y, self.action).__hash__()


    class Action(Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def _get_prev_state_actions(state):
        """ Returns set of states from which you can reach 'state' with one transition
        """
        return set([self._restrict_state(State(state.x+1, state.y)),
                self._restrict_state(State(state.x-1, state.y)),
                self._restrict_state(State(state.x, state.y+1)),
                self._restrict_state(State(state.x, state.y-1))])

    def _get_path_tree(self, state, l):

    def _restrict_state(self, state):
        return State(x = max(self.size-1, min(0, state.x))
                     y = max(self.size-1, min(0, state.y)))

    def _get_optimal_policy(self, variables):
        """ Returns a list of all possible paths that could have generated
            observation 'obs'.
        """
        raise NotImplementedError("Subclass implements")

    def _get_all_paths_for_obs(self, obs):
        """ Returns a tree containing all possible paths that could have generated
            observation 'obs'.
        """
        if obs not in self._precomp_paths.keys():
            if l > 0:
                self._precomp_paths[obs] = (state, )
                for next_state in self._get_prev_state_actions:
                    self._precomp_paths[obs] += (self._get_path_tree(Observation(next_state, l-1)), )
            else:
                self._precomp_paths[obs] = (state, )
        return PathTreeIterator(self._precomp_paths[obs], obs.path_len)

    class PathTreeIterator():
        """ Iterator for all paths
        """
        def __init__(self, paths, maxlen):
            self.paths = paths
            self.maxlen = maxlen

        def __iter__(self):
            self.indices = [0] * maxlen
            self.end = False
            return self

        def next(self):
            if self.end is True:
                raise StopIteration()
            path = list()
            root = self.paths
            nvals = list()
            for i in self.indices:
                path.append(root[0])
                nvals.append(len(root[1:]))
                root = root[i]
            for i, n in reversed(range(len(self.indices))):
                if self.indices[i] < nvals[i]:
                    self.indices[i] += 1
                    break
                else:
                    self.indices[i] = 0
            if max(self.indices) == 0:
                self.end = True
            return path

    def summary(path):
        """ Returns a summary observation of the full path
        """
        return Observation(path.states[0], len(path.states))

    def _prob_obs(self, obs, path):
        """ Returns the probability that 'path' would generate 'obs'.

        Parameters
        ----------
        obs : tuple (path x0, path y0, path length)
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        """
        # deterministic summary
        if summary(path) == obs:
            return 1.0
        return 0.0

    def _prob_path(self, path, policy, transfer):
        """ Returns the probability that 'path' would have been generated given 'policy'.

        Parameters
        ----------
        path : list of location tuples [(x0, y0), ..., (xn, yn)]
        policy : callable(state, action) -> p(action | state)
        transfer : callable(state, action, state') -> p(state' | state, action)
        """
        logp = 0
        # assume all start states equally probable
        for i in range(len(path.states)-1):
            act_i_prob = policy(path.states(i), path.actions(i))
            tra_i_prob = transfer(path.states(i), path.actions(i), path.states(i+1))
            logp += np.log(act_i_prob) + np.log(tra_i_prob)
        return np.exp(logp)


    def calculateDiscrepancy(observations1, observations2):
        features1 = [self._n_visits_to_loc(i, observations1) for i in range(self.n_init_locs)]
        features2 = [self._n_visits_to_loc(i, observations2) for i in range(self.n_init_locs)]
        disc = 0.0
        for f1, f2 in zip(features1, features2):
            disc += (f1 - f2) ** 2
        return disc

    def _n_visits_to_loc(idx, a):
        vals = []
        for ai in a:
            if ai[0] == idx:
                vals.append(ai[1])
        if len(vals) > 0:
            ret = np.mean(vals)
        else:
            ret = 0
        log.debug("visits({},{})={}".format(idx, a, ret))
        return np.atleast_1d(ret)


    def old_elfi_model():
        prior_means = [(bounds[i][1] - bounds[i][0])/2.0 for i in range(len(bounds))]
        prior_stds = [(bounds[i][1] - bounds[i][0])/4.0 for i in range(len(bounds))]
        W = [elfi.Prior("W{}".format(i),
                        "truncnorm",
                        (bounds[i][0] - prior_means[i]) / prior_stds[i],
                        (bounds[i][1] - prior_means[i]) / prior_stds[i],
                        prior_means[i],
                        prior_stds[i]
                        ) for i in range(len(weights))]
        Y = elfi.Simulator("MC", elfi.toold.vectorize(simulator), *W, observed=y)
        S = [elfi.Summary("S{}".format(i), means[i], Y) for i in range(n_init_locs)]
        d = elfi.Discrepancy("d", distance, *S)

        input_dim = len(weights)
        model = GPyModel(input_dim, bounds, kernel_var=1.0, kernel_scale=1.0, noise_var=0.01)
        acq = elfi.BolfiAcquisition(model,
                               exploration_rate=2.5,
                               n_samples=n_sim)
        bolfi = elfi.BOLFI(d, W,
                      batch_size=n_batch,
                      model=model,
                      acquisition=acq,
                      client=client,
                      n_opt_iters=10)

        result = bolfi.infer()
        log.info("MAP at {}".format(result.MAP))
        return result


    def old_do_inference_grid_world(client, outdir):
        np.random.seed(1234)
        bounds = (
            (-0.100, 0.0),
            (-0.100, 0.0)
            )
        weights = [
            -0.010,
            -0.050,
        ]
        n_sim = 200
        n_batch = 2
        n_rep = 5
        for i in range(n_rep):
            est = grid_world(client,
                             weights,
                             bounds,
                             seed=np.random.randint(10000),
                             n_sim=n_sim,
                             n_batch=n_batch)
            plot_2D_estimate(est, weights, bounds, outdir, i)


    def old_plot_2D_estimate(est, weights, bounds, outdir, idx):
        delta = 0.001
        X, Y, Z = eval_2d_mesh(bounds[0][0], bounds[1][0],
                               bounds[0][1], bounds[1][1],
                               200, 200, est.pdf)
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, 10)
        plt.xlabel("weight_0")
        plt.ylabel("weight_1")
        plt.scatter([weights[0]], [weights[1]], c="r", marker="o")
        plt.scatter([est.MAP[0]], [est.MAP[1]], c="b", marker="s")
        plt.savefig("{}/post_{:.2f}_{:.2f}_{}.png".format(outdir, weights[0], weights[1], idx))
        fig.clear()


    def old_eval_2d_mesh(xmin, ymin, xmax, ymax, nx, ny, eval_fun):
        """
            Evaluate 'eval_fun' at a grid defined by max and min
            values with number of points defined by 'nx' and 'ny'.
        """
        if xmin > xmax:
            raise ValueError("xmin (%.2f) was greater than"
                             "xmax (%.2f)" % (xmin, xmax))
        if ymin > ymax:
            raise ValueError("ymin (%.2f) was greater than"
                             "ymax (%.2f)" % (xmin, xmax))
        if nx < 1 or ny < 1:
            raise ValueError("nx (%.2f) or ny (%.2f) was less than 1" % (nx, ny))
        X = np.linspace(xmin, xmax, nx)
        lenx = len(X)
        Y = np.linspace(ymin, ymax, ny)
        leny = len(Y)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((leny, lenx))
        for i in range(leny):
            for j in range(lenx):
                Z[i][j] = eval_fun([X[i][j], Y[i][j]])
        return X, Y, Z


    def summary_fun(path):
        return (path[0][0], len(path)-1)


    def deduce_policy(paths):
        pol = dict()
        for path in paths:
            for i in range(1, len(path)-1):
                if path[i][2] == 0:
                    # move according to policy
                    s = (int(path[i][0]), int(path[i][1]))
                    if s not in pol.keys():
                        s_next = (int(path[i+1][0]), int(path[i+1][1]))
                        pol[s] = s_next
        return pol






    def test():
        test_post()
        test_meanf()
        test_distance()
        test_get_locations()

    if __name__ == "__main__":
        test()
        client = Client()
        dask.set_options(get=client.get)
        if len(sys.argv) != 3:
            print("grid_world.py <results_dir> <experiment_id>")
            sys.exit(-1)
        outdir = sys.argv[1]
        ex_id = sys.argv[2]
        resdir = outdir + "/" + ex_id
        os.makedirs(resdir)
        do_inference_grid_world(client, resdir)




class MenuSearchModel():
    """ Menu search model.
        From Chen et al. CHI 2016
        Used in Kangasrääsiö et al. CHI 2017
    """

    def __init__(self, variable_names):
        super(MenuSearchModel, self).__init__(variable_names)
        env = SearchEnvironment(
                    menu_type="semantic",
                    menu_groups=2,
                    menu_items_per_group=4,
                    semantic_levels=3,
                    gap_between_items=0.75,
                    prop_target_absent=0.1,
                    length_observations=True,
                    p_obs_len_cur=0.95,
                    p_obs_len_adj=0.89,
                    n_training_menus=10000)
        task = SearchTask(
                    env=env,
                    max_number_of_actions_per_session=20)
        self.rl = RLModel(
                    n_training_episodes=20000000,
                    n_episodes_per_epoch=10,
                    n_simulation_episodes=10000,
                    var_names=variable_names,
                    env=env,
                    task=task)
        self.used_discrepancy_features = {
            "00_task_completion_time": True,
            "01_task_completion_time_target_absent": False,
            "02_task_completion_time_target_present": False,
            "03_fixation_duration_target_absent": False,
            "04_fixation_duration_target_present": False,
            "05_saccade_duration_target_absent": False,
            "06_saccade_duration_target_present": False,
            "07_number_of_saccades_target_absent": False,
            "08_number_of_saccades_target_present": False,
            "09_fixation_locations_target_absent": False,
            "10_fixation_locations_target_present": False,
            "11_length_of_skips_target_absent": False,
            "12_length_of_skips_target_present": False,
            "13_location_of_gaze_to_target": False,
            "14_proportion_of_gaze_to_target": False
            }
        logger.info("Used discrepancy features: {}"
                .format([v for v in used_discrepancy_features.keys() if used_discrepancy_features[v] is True]))


    def evaluateLikelihood(variables, observations, random_state=None):
        raise NotImplementedError("Very difficult to evaluate.")

    def simulateObservations(variables, random_state):
        raise NotImplementedError("Subclass implements")

    def calculateDiscrepancy(observations1, observations2):
        features1 = feature_extraction(observations1)
        features2 = feature_extraction(observations1)
        discrepancy = Discrepancy(used_discrepancy_features=used_discrepancy_features)

    def getObservationDataset(menu_type="Semantic",
                              allowed_users=list(),
                              excluded_users=list(),
                              trials_per_user_present=1,
                              trials_per_user_absent=1):
        """ Returns the Bailly dataset as an observation.
        """
        dataset = BaillyData(menu_type,
                           allowed_users,
                           excluded_users,
                           trials_per_user_present,
                           trials_per_user_absent)
        return dataset.get()


    def old_inference_task():
        """Returns a complete Menu model in inference task

        Returns
        -------
        InferenceTask
        """
        logger.info("Constructing ELFI model..")
        itask = InferenceTask()
        variables = list()
        bounds = list()
        var_names = list()
        for var in inf_vars:
            if var == "focus_dur":
                v = elfi.Prior("focus_duration_100ms", "uniform", 1, 4, inference_task=itask)
                b = (1, 4)
            elif var == "recall_prob":
                v = elfi.Prior("menu_recall_probability", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            elif var == "semantic_obs":
                v = elfi.Prior("prob_obs_adjacent", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            elif var == "selection_delay":
                v = elfi.Prior("selection_delay_s", "uniform", 0, 1, inference_task=itask)
                b = (0, 1)
            else:
                assert False
            name = v.name
            logger.info("Added variable {}".format(name))
            var_names.append(name)
            variables.append(v)
            bounds.append(b)

        itask.parameters = variables
        itask.bounds = bounds
        logger.info("ELFI model done")
        return itask



def do_inference(inference_task, n_surrogate_samples=10, batch_size=1):
    acquisition = None  #default
    bounds = inference_task.bounds
    client = None  #default
    store = None # elfi.storage.DictListStore()
    kernel_class = "GPy.kern.RBF"
    noise_var = 0.05
    model = GPyModel(input_dim=len(bounds),
                    bounds=bounds,
                    kernel_class=eval(kernel_class),
                    kernel_var=0.05,
                    kernel_scale=0.1,
                    noise_var=noise_var,
                    optimizer="scg",
                    max_opt_iters=50)
    method = BOLFI(distance_node=inference_task.discrepancy,
                    parameter_nodes=inference_task.parameters,
                    batch_size=batch_size,
                    store=None,
                    model=model,
                    acquisition=acquisition,
                    sync=False,
                    bounds=bounds,
                    client=client,
                    n_surrogate_samples=n_surrogate_samples)
    posterior = method.infer()
    return posterior

def store_posterior(posterior, filename="out.json"):
    # hack
    assert type(posterior) is BolfiPosterior, type(posterior)
    model = posterior.model
    data = {
        "X_params": model.gp.X.tolist(),
        "Y_disc": model.gp.Y.tolist(),
        "kernel_class": model.kernel_class.__name__,
        "kernel_var": float(model.gp.kern.variance.values),
        "kernel_scale": float(model.gp.kern.lengthscale.values),
        "noise_var": model.noise_var,
        "threshold": posterior.threshold,
        "bounds": model.bounds,
        "ML": posterior.ML.tolist(),
        "ML_val": float(posterior.ML_val),
        "MAP": posterior.MAP.tolist(),
        "MAP_val": float(posterior.MAP_val),
        "optimizer": model.optimizer,
        "max_opt_iters": model.max_opt_iters
        }
    if filename is not None:
       f = open(filename, "w")
       json.dump(data, f)
       f.close()
    else:
        print("-----POSTERIOR-----")
        print(json.dumps(data))
        print("-------------------")
    logger.info("Stored compressed posterior to {}".format(filename))

def load_posterior(filename="out.json"):
    # hack
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    bounds = data["bounds"]
    kernel_class = eval("GPy.kern.{}".format(data["kernel_class"]))
    kernel_var = data["kernel_var"]
    kernel_scale = data["kernel_scale"]
    noise_var = data["noise_var"]
    optimizer = data["optimizer"]
    max_opt_iters = data["max_opt_iters"]
    model = GPyModel(input_dim=len(bounds),
                    bounds=bounds,
                    optimizer=optimizer,
                    max_opt_iters=max_opt_iters)
    model.set_kernel(kernel_class=kernel_class, kernel_var=kernel_var, kernel_scale=kernel_scale)
    X = np.atleast_2d(data["X_params"])
    Y = np.atleast_2d(data["Y_disc"])
    model._fit_gp(X, Y)
    posterior = BolfiPosterior(model, data["threshold"])
    posterior.ML = np.atleast_1d(data["ML"])
    posterior.ML_val = data["ML_val"]
    posterior.MAP = np.atleast_1d(data["MAP"])
    posterior.MAP_val = data["MAP_val"]
    return posterior



def test_distance():
    y = (np.array([[1.5]]), np.array([[3.0]]))
    x = (np.array([[2.5]]), np.array([[3.0]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[1.0]])+1))
    x = (np.array([[3.5]]), np.array([[3.0]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[4.0]])+1))
    x = (np.array([[1.4]]), np.array([[3.1]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[0.02]])+1))





if __name__ == "__main__":
    logging_setup()
    disable_pybrain_warnings()
    logger.info("Start")
    variables = ["focus_dur"]
    n_training_episodes = 1000
    n_surrogate_samples = 10
    batch_size = 5
    filename = "out2.json"
    task = inference_task(variables, n_training_episodes)
    posterior = do_inference(task, n_surrogate_samples, batch_size)
    store_posterior(posterior, None)
    p2 = load_posterior(filename)
    logger.info("End")
