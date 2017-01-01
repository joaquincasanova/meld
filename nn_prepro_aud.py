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
                              write_inverse_operator, apply_inverse_epochs)

###############################################################################
# Setup for reading the raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)

raw = mne.io.read_raw_fif(raw_fname, add_eeg_ref=False, preload=True)

raw.set_eeg_reference()  # set EEG average reference

baseline = (None, 0)  # means from the first instant to t = 0

reject = dict(mag=4e-12, eog=150e-6)

events = mne.read_events(event_fname)

picks = mne.pick_types(raw.info, meg='mag', eeg=True, eog=True,
                       exclude='bads') #for simplicity ignore grad channels

raw.rename_channels(mapping={'EOG 061': 'EOG'})

event_id = {'left/auditory': 1, 'right/auditory': 2,
            'left/visual': 3, 'right/visual': 4}

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, reject=reject, add_eeg_ref=False, preload=True)

epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

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

noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'])

###############################################################################
# Inverse modeling: MNE/dSPM on evoked and raw data
# -------------------------------------------------

# Read the forward solution and compute the inverse operator

fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-5-fwd.fif'
fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Restrict forward solution as necessary
fwd = mne.pick_types_forward(fwd, meg=True, eeg=True)

# make an inverse operator
info = epochs.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

write_inverse_operator('sample_audvis-meg-oct-5-inv.fif',
                       inverse_operator)

###############################################################################
# Compute inverse solution
# ------------------------

method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                    method=method, pick_ori=None)
#stc.save('sample_audvis-source-epochs')

total_batch_size = len(stc)#number of events. we'll consider each event an example.

n_steps=meg_data.shape[2]

dipole=stc[0]._data
for i in range(1,total_batch_size):
    dipole=np.dstack((dipole,stc[i]._data))
#pxn_stepsxbatchsize
qtrue_all,p=meas_class.scale_dipole(dipole)
#bxnxp


#meas_meg_in n_stepsxmxbatchsize
#meas_eeg_in n_stepsxmxbatchsize
meas_dims=[11,11]
print "Image grid dimensions: ", meas_dims
tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
tf_meas.scale()
tf_meas.interp()
tf_meas.reshape()
#tf_meas.plot(1)
meas_img_all = tf_meas.meas_img
m = tf_meas.m
