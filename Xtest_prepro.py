import numpy as np
from numpy import matlib
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time
#import nn_prepro
import prepro_class

def pred_obs(guess,locate,name,num,nummax,ax=None,fig=None):
    #print guess.shape
    if num is 0:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    markers = ['ob','or','ok','og']

    for l in range(0,locate):
        
        z = np.squeeze(guess[:,:,2+l*3]).reshape([-1,])
        y = np.squeeze(guess[:,:,1+l*3]).reshape([-1,])
        x = np.squeeze(guess[:,:,0+l*3]).reshape([-1,])
        #print z.shape
        if l is 0:
            ax.plot(x, y, z, markers[num],label=name)
        else:
            ax.plot(x, y, z, markers[num])

    if num is nummax-1:
        legend = plt.legend()
        plt.title('Location')
        #plt.savefig('facestreatmentsXYZT5.png')
        plt.show()
        
    #print num
    return ax, fig

pca=True
subsample=1
locate=95

for subject_id in ['aud',7,8]:
    for cnn in [True, False]:
        for rnn in [True, False]:
            if subject_id is 'aud':
                treats=['left/auditory', 'right/auditory', 'left/visual', 'right/visual']
            else:
                treats=['face/famous','scrambled','face/unfamiliar']
            nummax=len(treats)
            ax=None
            fig=None
            num=0
            prepro = prepro_class.prepro(selection='all',pca=pca,subsample=1,justdims=True,cnn=cnn,locate=locate,treat=None,rnn=rnn,Wt=None)
            if subject_id is 'aud':
                prepro.aud_dataset()
            else:
                prepro.faces_dataset(subject_id)

            Wt = prepro.Wt

            for treat in treats:
                prepro = prepro_class.prepro(selection='all',pca=pca,subsample=1,justdims=True,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                if subject_id is 'aud':
                    prepro.aud_dataset()
                else:
                    prepro.faces_dataset(subject_id)        

                test, val, batch_list, batches = prepro_class.ttv(prepro.total_batch_size,.2,.1,.1,rand_test=True)

                prepro = prepro_class.prepro(selection=test,pca=pca,subsample=1,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                if subject_id is 'aud':
                    prepro.aud_dataset()
                else:
                    prepro.faces_dataset(subject_id)
                meas_img_test = prepro.meas_img_all
                qtrue_test = prepro.qtrue_all
                p=prepro.p
                m=prepro.m

                ax,fig=pred_obs(qtrue_test,locate,treat,num,nummax,ax=ax,fig=fig)
                num+=1
