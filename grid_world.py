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

from distributed import Client
from functools import partial

import logging
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

def post(x):
    ret = np.array(eval(x), dtype=float)
    log.debug("sim: {}".format(ret))
    return ret

def test_post():
    out = "[[1,2], [1,3], [3,2], [2,3], [3,4]]"
    ret = np.array([[1,2], [1,3], [3,2], [2,3], [3,4]])
    np.testing.assert_array_equal(post(out), ret)

def meanf(idx, a):
    vals = []
    for ai in a:
        if ai[0] == idx:
            vals.append(ai[1])
    if len(vals) > 0:
        ret = np.mean(vals)
    else:
        ret = 0
    log.debug("mean({},{})={}".format(idx, a, ret))
    return np.atleast_1d(ret)

def test_meanf():
    a = np.array([[1,2], [1,3], [3,2], [2,3], [3,4]], dtype=float)
    np.testing.assert_array_equal(meanf(1, a), [2.5])
    np.testing.assert_array_equal(meanf(2, a), [3.0])
    np.testing.assert_array_equal(meanf(3, a), [3.0])
    np.testing.assert_array_equal(meanf(4, a), [0.0])

def distance(x, y):
    dist = np.zeros((x[0].shape[0], 1))
    for xi, yi in zip(x, y):
        dist += (xi - yi) ** 2
    dist = np.log(dist+1)
    log.debug("distance({},{})={}".format(x, y, dist))
    return dist

def test_distance():
    y = (np.array([[1.5]]), np.array([[3.0]]))
    x = (np.array([[2.5]]), np.array([[3.0]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[1.0]])+1))
    x = (np.array([[3.5]]), np.array([[3.0]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[4.0]])+1))
    x = (np.array([[1.4]]), np.array([[3.1]]))
    np.testing.assert_array_almost_equal(distance(x, y), np.log(np.array([[0.02]])+1))


def grid_world(client, weights=[0.01], bounds=((-0.1, 0.0),), seed=0, n_sim=100, n_batch=4):
    n_train = 500000
    n_obs = 1000
    size = 10
    prob_rnd_move = 0.1
    world_seed = np.random.randint(10000)
    n_init_locs = (size - 1) * 4
    output_type = 0  # summaries

    command = "cpp_models/build/toy_models grid_world {0} {1} {2} {3} {4} {seed} {5} {6}"
    for i in range(len(weights)):
        command += " {" + str(i+7) + "}"

    # simulate observations
    for i in range(1):
        print("True weights: {}".format(weights))
        cmd = command.format(size, prob_rnd_move, n_train, n_obs, output_type, world_seed, len(weights), *weights, seed=seed+i)
        y = Wrapper(cmd, post)()
        log.info("out: {}".format([meanf(i, y)[0] for i in range(int(max([yj[0] for yj in y])))]))

    do_inference_baseline(y, command, size, prob_rnd_move, n_train, world_seed, bounds)

    simulator = partial(Wrapper(command, post), size, prob_rnd_move, n_train, world_seed, len(weights))

    means = [elfi.tools.vectorize(partial(meanf, i)) for i in range(n_init_locs)]

    # Specify the graphical model
    #W = [elfi.Prior("W{}".format(i), "uniform", bounds[i][0], bounds[i][1]) for i in range(len(weights))]
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


def do_inference_grid_world(client, outdir):
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


def plot_2D_estimate(est, weights, bounds, outdir, idx):
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


def eval_2d_mesh(xmin, ymin, xmax, ymax, nx, ny, eval_fun):
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


def get_locations(bounds, delta):
    locs = [tuple()]
    for l, u in bounds:
        vals = np.arange(l, u, delta)
        newlocs = [l + (v,) for l in locs for v in vals]
        locs = newlocs
    return locs


def test_get_locations():
    l = get_locations(((0,5),), 1)
    vals = ((0,), (1,), (2,), (3,), (4,))
    for v in vals:
        assert v in l
    l = get_locations(((0,10),(10,20)), 5)
    vals = ((0, 10), (0, 15), (5, 10), (5, 15))
    for v in vals:
        assert v in l


def post2(x):
    ret = np.array(eval(x))
    log.debug("sim: {}".format(ret))
    return ret


def select_unique_paths(paths):
    s = set()
    for path in paths:
        s.add(tuple(path))
    return s


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


def restrict_loc(size, v):
    return max(size-1, min(0, v))


def get_path_tree(x, y, l, r):
    if (x, y, l) not in get_path_tree.precomp_paths.keys():
        if l > 0:
            get_path_tree.precomp_paths[(x, y, l)] = ((x, y),
                                                      get_path_tree(r(x-1), y, l-1, r),
                                                      get_path_tree(r(x+1), y, l-1, r),
                                                      get_path_tree(x, r(y-1), l-1, r),
                                                      get_path_tree(x, r(y+1), l-1, r))
        else:
            get_path_tree.precomp_paths[(x, y, l)] = ((x, y), )
    return get_path_tree.precomp_paths[(x, y, l)]

get_path_tree.precomp_paths = dict()

def compute_paths(root, path, data):
    # compute path probabilities recursively by traversing the tree
    path.append(root[0])
    if len(root) == 1:
        # leaf
        calc_path_prob(path, data)
    if len(root) == 5:
        # non-leaf
        for c in root[1:]:
            compute_paths(c, path[:], data)


def calc_path_prob(path, data):
    if path[-1][0] != data["size"]/2 or path[-1][1] != data["size"]/2:
        # not correct end state
        return
    logp = 0
    assert len(path) == data["y"][1]
    # transition probabilities
    for i in range(y[1]-1):
        nxt = data["policy"][(path[i][0], path[i][1])]
        if nxt[0] == path[i+1][0] and nxt[1] == path[i+1][1]:
            logp += data["logp_det"]
        else:
            logp += data["logp_rnd"]
    p = np.exp(logp)
    #print(p)
    data["prob"] += p


def compute_observation_probability(y, policy, size, prob_rnd_move):
    totprob = 1.0
    for yi in y:
        prob = 0.0
        l = yi[1]
        x0 = yi[2]
        y0 = yi[3]
        p_rnd = prob_rnd_move / 3.0  # probability of taking each individual obs rnd move
        data = {
                "prob": 0.0,
                "y": yi,
                "policy": policy,
                "size": size,
                "logp_rnd": np.log(p_rnd),
                "logp_det": np.log(1.0 - p_rnd)
                }
        path_tree = get_path_tree(x0, y0, l, partial(restrict_loc, size))
        compute_paths(path_tree, list(), data)
        print("- Prob for {} is {}".format(yi, data["prob"]))
        totprob *= data["prob"]
    return totprob


def do_inference_baseline(y, command, size, prob_rnd_move, n_train, world_seed, bounds):
    delta = 0.05
    locs = get_locations(bounds, delta)
    print("Solving baseline in {} locations".format(len(locs)))
    seed = 1234
    n_all = 10000  # assume this contains a policy move for all grid cells
    output_type = 1  # full trace
    maxprob = -1
    maxloc = None
    for i, loc in enumerate(locs):
        print("Solving baseline in {}".format(loc))
        cmd = command.format(size, prob_rnd_move, n_train, n_all, output_type, world_seed, len(loc), *loc, seed=seed+i)
        paths = Wrapper(cmd, post2)()
        #print("Observed paths {}".format(paths))
        unique_paths = select_unique_paths(paths)
        policy = deduce_policy(unique_paths)
        prob = compute_observation_probability(y, policy, size, prob_rnd_move)
        print("Total probability is {}".format(prob))
        if prob > maxprob:
            maxprob = prob
            maxloc = loc
    print("Maximum probability ({}) location at {}".format(maxprob, maxloc))


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

