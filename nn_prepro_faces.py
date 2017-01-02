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


study_path = '/home/jcasa/mne_data/openfmri'
subjects_dir = os.path.join(study_path, 'subjects')
meg_dir = os.path.join(study_path, 'MEG')
os.environ["SUBJECTS_DIR"] = subjects_dir
spacing = 'oct5'
mindist = 5

for subject_id in [7]:
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
    total_batch_size = len(stc)#number of events. we'll consider each event an example.

    n_eeg = epochs_eeg.get_data().shape[1]

    eeg_xyz = epochs_eeg.info['chs'][0]['loc'][:3].reshape([1,3])
    eeg_data = np.array(epochs_eeg.get_data())#batch_sizexmxn_steps

    for i in range(1,n_eeg):
        eeg_xyz=np.vstack((eeg_xyz,epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3])))

    n_meg = epochs_meg.get_data().shape[1]

    meg_xyz = epochs_meg.info['chs'][0]['loc'][:3].reshape([1,3])
    meg_data = np.array(epochs_meg.get_data())#batch_sizexmxn_steps

    for i in range(1,n_meg):
        meg_xyz=np.vstack((meg_xyz,epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3])))
    n_steps=meg_data.shape[2]

#    dipole=stc[0]._data
#    for i in range(1,total_batch_size):
#        dipole=np.dstack((dipole,stc[i]._data))
#    #pxn_stepsxbatchsize
#    qtrue_all,p=meas_class.scale_dipole(dipole)
#    #bxnxp
    
    meas_dims=[11,11]
    print "Image grid dimensions: ", meas_dims
    tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
    tf_meas.scale()
    tf_meas.interp()
    tf_meas.reshape()
    #tf_meas.plot(1)
    meas_img_all = tf_meas.meas_img
    m = tf_meas.m
 
