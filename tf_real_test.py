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
test_size = total_batch_size % 200

n_steps=meg_data.shape[2]

dipole=stc[0]._data
for i in range(1,total_batch_size):
    dipole=np.dstack((dipole,stc[i]._data))
#pxn_stepsxbatchsize
qtrue_all,p=meas_class.scale_dipole(dipole)
#bxnxp


#meas_meg_in n_stepsxmxbatchsize
#meas_eeg_in n_stepsxmxbatchsize
meas_dims=[13,13]
print "Image grid dimensions: ", meas_dims
tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
tf_meas.scale()
tf_meas.interp()
tf_meas.reshape()
#tf_meas.plot(1)
meas_img_all = tf_meas.meas_img
m = tf_meas.m

fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

with open('./nn_real.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for cost in ['cross','rmse']:
        for cost_step in ['last','all']:
            for learning_rate in [0.005]:
                for batches in [5]:
                    for dropout in [.99]:
                        for per_batch in [500]:
                            for batch_size in [(total_batch_size-test_size)/batches]:
                                for k_conv in [3]:
                                    for n_conv1 in [2, 5, 10]:
                                        for n_conv2 in [2, 5, 10]:
                                            for n_layer in [2, 3]:
                                                for n_lstm in [100, 300, 1000]:
                                                    test_size=batch_size
                                                    
                                                    n_chan_in=2
                                                    k_pool=1
                                                    n_out=p
                                                    n_in=meas_dims[0]*meas_dims[1]*2
                                                    n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2

                                                    #pick a first batch of batch_size
                                                    meas_img = meas_img_all[0:(batch_size),:,:,:,:]
                                                    qtrue = qtrue_all[0:(batch_size),:,:]
                                                    batch_num = 0

                                                    cnn_rnn=tf_class.tf_meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cost_func=cost,cost_time=cost_step)
                                                    tf.reset_default_graph()
                                                    cnn_rnn.network()

                                                    with tf.Session() as session:

                                                        session.run(cnn_rnn.init_step)

                                                        for step in range(0,per_batch*batches):

                                                            _ , guess,ce,acc,err,ce_l,acc_l,err_l  = session.run([cnn_rnn.train_step, cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                 feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout})

                                                            if step % 10==0:
                                                                print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                                                writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})
                                                            if step % per_batch ==0 and step!=0:#generate a new batch
                                                                batch_num = batch_num+1
                                                                #pick a nth batch of batch_size
                                                                print "New batch"
                                                                meas_img = meas_img_all[batch_size*batch_num:(batch_size*(batch_num+1)),:,:,:,:]
                                                                qtrue = qtrue_all[batch_size*batch_num:(batch_size*(batch_num+1)),:,:]

                                                        #test batch
                                                        meas_img = meas_img_all[batch_size*(batch_num+1):,:,:,:,:]
                                                        qtrue = qtrue_all[batch_size*(batch_num+1):,:,:]
                                                        guess,ce,acc,err,ce_l,acc_l,err_l = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                    feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout})
                                                        print "Test Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l

                                                        writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})

    csvfile.close()
