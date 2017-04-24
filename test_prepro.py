import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
import nn_prepro

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
locate=95

subject_id=7

if subject_id is 'aud':
    treats=[None]#, 'left/auditory', 'right/auditory', 'left/visual', 'right/visual']
else:
    treats=[None]#, 'face/famous','scrambled','face/unfamiliar']
nummax=len(treats)
ax=None
fig=None
num=0
for treat in treats:
    if subject_id is 'aud':
        meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=treat)
    else:
        meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=treat)
