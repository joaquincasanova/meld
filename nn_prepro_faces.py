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

subject_id=5
subject = "sub%03d" % subject_id
print("processing %s" % subject)

invname = '%s-meg-%s-inv.fif' % (subject,spacing)
invpath = os.path.join(os.path.join(meg_dir,subject),invname) 
fwdname = '%s-meg-%s-fwd.fif' % (subject,spacing)
fwdpath = os.path.join(os.path.join(meg_dir,subject),fwdname) 
eponame = '%s-epo.fif' % (subject) 
epopath = os.path.join(os.path.join(meg_dir,subject),eponame)

picks = mne.pick_types(raw.info, meg='mag', eeg=True, eog=True,
                       exclude='bads') #for simplicity ignore grad channels

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

dipole=stc[0]._data
for i in range(1,total_batch_size):
    dipole=np.dstack((dipole,stc[i]._data))
#pxn_stepsxbatchsize
qtrue_all,p=meas_class.scale_dipole(dipole)
