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
                              write_inverse_operator, apply_inverse_epochs,
                              read_inverse_operator)
def aud_dataset(selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=False):
    ###############################################################################
    # Setup for reading the raw data
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)
    subject='sample'
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
 
    if justdims is True:
        meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
        return meas_dims, m, p, n_steps, total_batch_size
    else:
        meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
        return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 

def faces_dataset(subject_id,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=False):
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
    epochs = mne.read_epochs(epopath,verbose=False)

    epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
    epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

    fwd = mne.read_forward_solution(fwdpath,verbose=False)
    inv = read_inverse_operator(invpath,verbose=False)
    method = "MNE"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse_epochs(epochs, inv, lambda2,
                               method=method, pick_ori=None,verbose=False)
    if justdims is True:
        meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
        return meas_dims, m, p, n_steps, total_batch_size
    else:
        meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
        return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 


def prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=False):
    if cnn is True:
        if justdims is True:
            total_batch_size = len(stc)#number of events. we'll consider each event an example.
            if locate is True:
                p=3
            else:
                p = stc[0]._data.shape[0]
            n_steps = stc[0]._data.shape[1]
            meas_dims=[11,11]
            m = meas_dims[0]*meas_dims[1]
            del stc, epochs, epochs_eeg, epochs_meg
            if locate is True:
                p=3
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
            elif pca is False:
                tf_meas.scale()
            else:
                pass
            
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
                qtrue_all, p = location(stc,subject,selection=selection)
            else:
                #pxn_stepsxbatchsize
                qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
                #bxnxp
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 
    else:
        if justdims is True:
            total_batch_size = len(stc)#number of events. we'll consider each event an example.
            if locate is True:
                p=3
            else:
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
            elif pca is False:
                tf_meas.scale()
            else:
                pass

            tf_meas.stack_reshape()
            meas_img_all = tf_meas.meas_stack
            m = tf_meas.m

            if selection is 'all':
                dipole=np.array([stc[i]._data for i in range(0,len(stc))]).transpose((1,2,0))
            else:
                dipole=np.array([stc[i]._data for i in selection]).transpose((1,2,0))

            if locate==True:
                qtrue_all, p = location(stc,subject,selection=selection)
            else:
                #pxn_stepsxbatchsize
                qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
                #bxnxp
            del stc, epochs, epochs_eeg, epochs_meg
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size 

def location(stc,subject,selection='all'):
    if selection is 'all':
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(stc),ns,3))
        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(stc)):
            #max location (index)
            mxloc = np.argmax(stc[s].data,axis=0)#1xn_steps
            #what hemisphere?
            hemi = np.where(mxloc<nd/2,0,1)
            mxvtx_long =vtx_long[mxloc]
            
            #for m in range(0,ns):
            #    mxvtx = vtx[hemi[m]][mxloc[m]-hemi[m]*hem0]#1xn_steps
            loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False)
        qtrue_all = loc
        p=3
        return qtrue_all, p
    else:
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(selection),ns,3))
        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(selection)):
            #max location (index)
            mxloc = np.argmax(stc[s].data,axis=0)#1xn_steps
            #what hemisphere?
            hemi = np.where(mxloc<nd/2,0,1)
            mxvtx_long =vtx_long[mxloc]
            
            #for m in range(0,ns):
            #    mxvtx = vtx[hemi[m]][mxloc[m]-hemi[m]*hem0]#1xn_steps
            loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False)
        qtrue_all = loc
        p=3
        return qtrue_all, p

def merge_2(s1,s2,selection1='all',selection2='all',selection='all',pca=False,subsample=1,justdims=False,cnn=True,locate=False):
    subject_id = s1
    meas_dims, m, p, n_steps, batch_size1 = faces_dataset(subject_id,locate=True)
    meas_img1, qtrue1, meas_dims, m, p, n_steps, batch_size1 = faces_dataset(subject_id,selection=selection1,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)

    subject_id = s2
    meas_dims, m, p, n_steps, batch_size2 = faces_dataset(subject_id,locate=True)
    meas_img2, qtrue2, meas_dims, m, p, n_steps, batch_size2 = faces_dataset(subject_id,selection=selection2,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate)
    
    meas_img = np.vstack([meas_img1,meas_img2])
    qtrue = np.vstack([qtrue1,qtrue2])
    tot=meas_img.shape[0]
    perm=np.random.permutation(tot)
    meas_img = meas_img[perm,:,:,:,:]
    qtrue = qtrue[perm,:,:]

    meas_img = meas_img[selection,:,:,:,:]
    qtrue = qtrue[selection,:,:]

    return meas_img, qtrue, meas_dims, m, p, n_steps, len(selection) 
