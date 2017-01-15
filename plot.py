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

