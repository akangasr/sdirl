import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dask
import GPy
import elfi
from elfi.wrapper import Wrapper
from elfi.bo.gpy_model import GPyModel

from distributed import Client
from functools import partial

import logging
elfi_log = logging.getLogger("elfi")
elfi_log.setLevel(logging.INFO)
elfi_log.propagate = False
log = logging.getLogger("mountain_car")
log.setLevel(logging.INFO)
log.propagate = False

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
elfi_log.handlers = [ch]
log.handlers = [ch]


def mountain_car(client, val_min=0.0, val_max=2.0, val=1.0, seed=0, n_sim=20, n_batch=4):
    throttle_min = val_min
    throttle_max = val_max
    throttle = val
    n_train = 10000
    n_obs = 10

    command = "cpp_models/build/toy_models mountain_car {2:.2f} {0} {1} {seed}"

    def post(x):
        ret = np.atleast_2d(eval(x))
        log.debug("return {}".format(ret))
        return ret

    # simulate observations
    cmd = command.format(n_train, n_obs, throttle, seed=seed)
    y = Wrapper(cmd, post)()

    simulator = partial(Wrapper(command, post), n_train, n_obs)

    def meanf(idx, a):
        m = np.mean(a[:,:,idx], keepdims=True)
        log.debug("meanf({},{})={}".format(idx, a, m))
        return m

    loc = partial(meanf, 0)
    vel = partial(meanf, 1)

    def distance(x, y):
        loc_sim = x[0]
        vel_sim = x[1]
        loc_obs = y[0]
        vel_obs = y[1]
        d = abs(100*loc_sim - 100*loc_obs) ** 2 + abs(vel_sim - vel_obs) ** 2
        log.debug("distance({},{})={}".format(x, y, d))
        return d

    # Specify the graphical model
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


def do_inference_mountain_car(client, outdir):
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


def plot_estimates(estimates, location, locrange, outdir):
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

if __name__ == "__main__":
    client = Client()
    dask.set_options(get=client.get)
    if len(sys.argv) != 3:
        print("mountain_car.py <results_dir> <experiment_id>")
    outdir = sys.argv[1]
    ex_id = sys.argv[2]
    resdir = outdir + "/" + ex_id
    os.makedirs(resdir)
    do_inference_mountain_car(client, resdir)

