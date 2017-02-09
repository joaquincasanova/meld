import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
import nn_prepro
test = np.random.choice(879,1,replace=False)
test_id=7
pca=True
subsample=1
meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=True,locate=True)
z = np.squeeze(qtrue_test[0,:,2])
y = np.squeeze(qtrue_test[0,:,1])
x = np.squeeze(qtrue_test[0,:,0])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)

plt.show()
