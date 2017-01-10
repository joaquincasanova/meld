import os
import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import tf_class
import csv
import meas_class
import mne
from mne.datasets import sample
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, apply_inverse_epochs,read_inverse_operator)
def prepro(stc, epochs, epochs_eeg,epochs_meg,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=False):
    if cnn is True:
        if justdims is True:
            total_batch_size = len(stc)#number of events. we'll consider each event an example.
            p = stc[0]._data.shape[0]
            n_steps = stc[0]._data.shape[1]
            meas_dims=[11,11]
            m = meas_dims[0]*meas_dims[1]
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_dims, m, p, n_steps, total_batch_size
        else:
            if selection is 'all':
                total_batch_size = len(stc)#number of events. we'll consider each event an example.
            else:
                total_batch_size = len(selection)#number of events. we'll consider each event an example.

            n_eeg = epochs_eeg.get_data().shape[1]

            eeg_xyz=np.squeeze(np.array([epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

            n_meg = epochs_meg.get_data().shape[1]

            meg_xyz=np.squeeze(np.array([epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

            if selection is 'all':
                eeg_data = np.array(epochs_eeg.get_data())#batch_sizexmxn_steps
                meg_data = np.array(epochs_meg.get_data())#batch_sizexmxn_steps
            else:
                eeg_data = np.array(epochs_eeg.get_data())[selection,:,:]#batch_sizexmxn_steps
                meg_data = np.array(epochs_meg.get_data())[selection,:,:]#batch_sizexmxn_steps

            n_steps=meg_data.shape[2]

            meas_dims=[11,11]
            print "Image grid dimensions: ", meas_dims
            tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
            if pca is True:
                tf_meas.pca()
            else:
                tf_meas.scale()
            tf_meas.interp()
            tf_meas.reshape()
            #tf_meas.plot(1)
            meas_img_all = tf_meas.meas_img
            m = tf_meas.m

            if selection is 'all':
                dipole=np.array([stc[i]._data for i in range(0,len(stc))]).transpose((1,2,0))
            else:
                dipole=np.array([stc[i]._data for i in selection]).transpose((1,2,0)) 
            
            if locate==True:
                qtrue_all, p = location(stc)
            else:
                #pxn_stepsxbatchsize
                qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
                #bxnxp
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 
    else:
        if justdims is True:
            total_batch_size = len(stc)#number of events. we'll consider each event an example.
            p = stc[0]._data.shape[0]
            n_steps = stc[0]._data.shape[1]
            meas_dims=epochs.get_data().shape[1]
            m = meas_dims
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_dims, m, p, n_steps, total_batch_size
        else:
            if selection is 'all':
                total_batch_size = len(stc)#number of events. we'll consider each event an example.
            else:
                total_batch_size = len(selection)#number of events. we'll consider each event an example.

            n_eeg = epochs_eeg.get_data().shape[1]

            eeg_xyz=np.squeeze(np.array([epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

            n_meg = epochs_meg.get_data().shape[1]

            meg_xyz=np.squeeze(np.array([epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

            if selection is 'all':
                eeg_data = np.array(epochs_eeg.get_data())#batch_sizexmxn_steps
                meg_data = np.array(epochs_meg.get_data())#batch_sizexmxn_steps
            else:
                eeg_data = np.array(epochs_eeg.get_data())[selection,:,:]#batch_sizexmxn_steps
                meg_data = np.array(epochs_meg.get_data())[selection,:,:]#batch_sizexmxn_steps

            n_steps=meg_data.shape[2]

            meas_dims=n_eeg+n_meg
            print "Meas dims in: ", meas_dims
            tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
            if pca is True:
                tf_meas.pca()
            else:
                tf_meas.scale()
            tf_meas.stack_reshape()
            meas_img_all = tf_meas.meas_stack
            m = tf_meas.m

            if selection is 'all':
                dipole=np.array([stc[i]._data for i in range(0,len(stc))]).transpose((1,2,0))
            else:
                dipole=np.array([stc[i]._data for i in selection]).transpose((1,2,0))

            if locate==True:
                qtrue_all, p = location(stc,selection=selection)
            else:
                #pxn_stepsxbatchsize
                qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
                #bxnxp
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 

def location(stc):
    nd=stc[0].data.shape[0]
    ns=stc[0].data.shape[1]
    loc=np.zeros((len(stc),ns,3))
    vtx = stc[0].vertices
    hem0 = np.size(vtx[0])
    hem1 = np.size(vtx[1])
    for s in range(0,len(stc)):
        #max location (index)
        mxloc = np.argmax(stc[s].data,axis=0)#1xn_steps
        #what hemisphere?
        hemi = np.where(mxloc<nd/2,0,1)
        for m in range(0,ns):
            mxvtx = vtx[hemi[m]][mxloc[m]-hemi[m]*hem0]#1xn_steps
            loc[s,ns,:] = mne.vertex_to_mni(mxvtx,hemi[m],subject,verbose=False)
    qtrue_all = loc
    p=3
    return qtrue_all, p

def location(stc,selection='all'):
    if selection is 'all':
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(stc),ns,3))
        vtx = stc[0].vertices
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(stc)):
            #max location (index)
            mxloc = np.argmax(stc[s].data,axis=0)#1xn_steps
            #what hemisphere?
            hemi = np.where(mxloc<nd/2,0,1)
            for m in range(0,ns):
                mxvtx = vtx[hemi[m]][mxloc[m]-hemi[m]*hem0]#1xn_steps
                loc[s,m ,:] = mne.vertex_to_mni(mxvtx,hemi[m],subject,verbose=False)
        qtrue_all = loc
        p=3
        return qtrue_all, p
    else:
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(selection),ns,3))
        vtx = stc[0].vertices
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(selection)):
            #max location (index)
            mxloc = np.argmax(stc[selection[s]].data,axis=0)#1xn_steps
            #what hemisphere?
            hemi = np.where(mxloc<nd/2,0,1)
            for m in range(0,ns):
                mxvtx = vtx[hemi[m]][mxloc[m]-hemi[m]*hem0]#1xn_steps
                loc[s,m ,:] = mne.vertex_to_mni(mxvtx,hemi[m],subject,verbose=False)
        qtrue_all = loc
        p=3
        return qtrue_all, p

subject_id=7
selection=np.arange(10)
pca=False
subsample=1
justdims=False
cnn=False
locate=True
study_path = '/home/jcasa/mne_data/openfmri'
subjects_dir = os.path.join(study_path, 'subjects')
meg_dir = os.path.join(study_path, 'MEG')
os.environ["SUBJECTS_DIR"] = subjects_dir
spacing = 'oct5'
mindist = 5

subject = "sub%03d" % subject_id
print("processing %s" % subject)

invname = '%s-meg-%s-inv.fif' % (subject,spacing)
invpath = os.path.join(os.path.join(meg_dir,subject),invname) 
fwdname = '%s-meg-%s-fwd.fif' % (subject,spacing)
fwdpath = os.path.join(os.path.join(meg_dir,subject),fwdname) 
eponame = '%s-epo.fif' % (subject) 
epopath = os.path.join(os.path.join(meg_dir,subject),eponame)
epochs = mne.read_epochs(epopath)

epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

fwd = mne.read_forward_solution(fwdpath)
inv = read_inverse_operator(invpath)
method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse_epochs(epochs, inv, lambda2,
                    method=method, pick_ori=None)
if justdims is True:
    meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
else:
    meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)


