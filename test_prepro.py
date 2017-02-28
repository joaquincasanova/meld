import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
import Xnn_prepro as nn_prepro

def pred_obs(guess,locate,name,num,nummax,ax=None,fig=None):
    print guess.shape
    if num is 0:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    markers = ['ob','or','ok','og']

    for l in range(0,locate):
        
        z = np.squeeze(guess[:,:,2+l*3]).reshape([-1,])
        y = np.squeeze(guess[:,:,1+l*3]).reshape([-1,])
        x = np.squeeze(guess[:,:,0+l*3]).reshape([-1,])
        print z.shape
        if l is 0:
            ax.plot(x, y, z, markers[num],label=name)
        else:
            ax.plot(x, y, z, markers[num])

    if num is nummax-1:
        legend = plt.legend()
        plt.title('Location')
        #plt.savefig('facestreatmentsXYZT5.png')
        plt.show()
        
    print num
    return ax, fig

pca=True
subsample=1
pca=True
cnn=True
locate=1

subject_id='aud'

if subject_id is 'aud':
    treats=['left/auditory', 'right/auditory', 'left/visual', 'right/visual']
else:
    treats=['face/famous','scrambled','face/unfamiliar']
nummax=len(treats)
ax=None
fig=None
num=0
for treat in treats:
    if subject_id is 'aud':
        meas_dims, m, p, n_steps, total_batch_size = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=treat)
    else:
        meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=treat)

    test, val, batch_list, batches = nn_prepro.ttv(total_batch_size,.2,.1,.1,rand_test=True)
    #test = 'all'
    if subject_id is 'aud':
        meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.aud_dataset(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat)
    else:
        meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(subject_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat)
    print qtrue_test.shape

    ax,fig=pred_obs(qtrue_test,locate,treat,num,nummax,ax=ax,fig=fig)
    num+=1
