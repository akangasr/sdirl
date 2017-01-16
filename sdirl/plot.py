import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm

with open("test.txt") as file:
    array2d = [[float(digit) for digit in line.split()] for line in file]

x = range(len(array2d))
y = range(len(array2d[0]))
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, np.atleast_2d(array2d), cmap = cm.gray)
plt.show()



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

