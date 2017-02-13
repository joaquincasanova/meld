import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
import nn_prepro

pca=True
subsample=1
pca=True
cnn=True
locate=True

subject_id='aud'
if subject_id is 'aud':
    meas_dims, m, p, n_steps, total_batch_size = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate)
else:
    meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate)

test, val, batch_list, batches = nn_prepro.ttv(total_batch_size,.2,.1,.1,rand_test=True)

if subject_id is 'aud':
    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.aud_dataset(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
else:
    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(subject_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)

z = np.squeeze(qtrue_test[0,:,2])
y = np.squeeze(qtrue_test[0,:,1])
x = np.squeeze(qtrue_test[0,:,0])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)

plt.show()
