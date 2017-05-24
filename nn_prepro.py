import os
import numpy as np
from numpy import matlib
import sphere
import dipole_class_rat
import csv
import meas_class
import mne
import sphere
from mne.datasets import sample
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, apply_inverse_epochs,
                              read_inverse_operator)
import pickle

def rat_real(stim='Tones',selection='all',pca=True,subsample=1,justdims=True,cnn=False,locate=True,treat=None,rnn=False,Wt=None):
    #print 'selection ',selection
    ecog_thresh = 1e-5
    if stim=='Tones':
        name = '/home/jcasa/meld/code/python/rattest/processed/ECOG_MEG_Tones.grouped.pickle'
    elif stim=='P1':
        name = '/home/jcasa/meld/code/python/rattest/processed/ECOG_MEG_P1.grouped.pickle'
    elif stim=='P0':
        name = '/home/jcasa/meld/code/python/rattest/processed/MEG_ECOG.grouped.pickle'
    with open(name, 'r') as f:
        b = pickle.load(f)
    ecog_data=np.transpose(np.array(b["ECoG_average"]),(1,2,0))#pxnxb - for dipole scaling
    eeg_data=np.transpose(np.array(b["ECoG_average"]),(0,1,2))
    ecog_data[abs(ecog_data)>ecog_thresh]=100e-9#A
    #eeg_data[abs(eeg_data)>ecog_thresh]=100e-6#V
    meg_data=np.transpose(np.array(b["MEG_average"]),(0,1,2))#bxmxn - for meas_class pca formatting
    fs_MEG=b["fs_MEG"]
    fs_ECoG=b["fs_ECoG"]
    flag=b["flag"]
    n_treat=b["n_treat"]
    treatments=b["treatments"]
    meg_xyz=b["meg_xyz"]/1000.#mx3
    ecog_xyz=b["ecog_xyz"]/1000.#in meters
    n_chan_in=1
    n_steps = meg_data.shape[2]
    total_batch_size = ecog_data.shape[2]
    m = meg_data.shape[1]
    p = ecog_data.shape[0]
    meas_dims=m
    #print 'n_steps', n_steps, 'total_batch_size', total_batch_size, 'm', m, 'p', p
    #print 'MEG array: ',meg_data.shape
    #print 'ECOG array: ',ecog_data.shape
    #print 'fake EEG array: ',eeg_data.shape
    if pca:
        tf_meas = meas_class.meas(meg_data,meg_xyz,eeg_data,ecog_xyz, meas_dims, n_steps, total_batch_size)
        Wt=tf_meas.pca()
        tf_meas.stack_reshape(n_chan_in=1)#ignore ecog - just a placeholder
        meas_img_all = tf_meas.meas_stack
    else:
        meas_img_all = np.transpose(meg_data,(0,2,1))
                          
    #scale dipoles ~ ecog
    #print 'Locate: ',locate
    if rnn:
        if locate is False:
            qtrue_all, p = meas_class.scale_dipoleXYZT_OH(ecog_data,subsample=subsample)
        else:
            p=3
            qtrue_all = location_rat_XYZT(locate,total_batch_size,n_steps,p,ecog_data, ecog_xyz)
    else:
        if locate is False:
            qtrue_all, p = meas_class.scale_dipole(ecog_data,subsample=subsample)
        else:
            p=3
            qtrue_all = location_rat(locate,total_batch_size,n_steps,p,ecog_data, ecog_xyz)

    if justdims is True:
        return meas_dims, m, p, n_steps, total_batch_size, Wt
    else:
        #print 'meas_img_all ',meas_img_all.shape
        #print 'qtrue_all ',qtrue_all.shape
        
        if selection is 'all':
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 
        else:
            return meas_img_all[selection,:,:], qtrue_all[selection,:,:], meas_dims, m, p, n_steps, np.size(selection), Wt 

    
def rat_synth(total_batch_size,delT,n_steps,meas_dims,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,treat=None,rnn=True,Wt=None):
    
    if selection is 'all':
        batch_size=total_batch_size
    else:
        batch_size=len(selection)
    subject='rat'
    print dipole_dims, "Dipole dims"
    print batch_size, "Batch size"
    instance = dipole_class_rat.dipole(delT, batch_size, n_steps,
                      2, meas_dims, dipole_dims,
                      orient=orient, noise_flag=noise_flag,
                      dipole_xyz=dipole_xyz, meas_xyz=meas_xyz,pca=pca)
    instance.batch_sequence_gen()

    meg_data=instance.meg_data
    eeg_data=instance.eeg_data
    dipole=instance.qtrue
    m=instance.m
    p=instance.p
    #print p, dipole.shape, "Dipoles (rat_synth)"
    meg_xyz=instance.meg_xyz
    assert meg_xyz.shape[0]==m and meg_xyz.shape[1]==3, meg_xyz.shape
    eeg_xyz=instance.eeg_xyz
    assert eeg_xyz.shape[0]==m and eeg_xyz.shape[1]==3, eeg_xyz.shape

    dipole_xyz=instance.dipole_xyz
    assert dipole_xyz.shape[0]==p and dipole_xyz.shape[1]==3, dipole_xyz.shape
    n_steps=instance.n_steps
    batch_size=instance.batch_size
    meas_img_all, qtrue_all, meas_dims, m, p, n_steps, batch_size, Wt = rat_prepro(n_chan_in,dipole,dipole_xyz,meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, batch_size,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
    if justdims is True:
        return meas_dims, m, p, n_steps, batch_size, Wt
    else:
        return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, batch_size, Wt 

def aud_dataset(selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,treat=None,rnn=True,Wt=None):
    print 'Treat: ', treat
    ###############################################################################
    # Setup for reading the raw data
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)
    subject='sample'
    raw = mne.io.read_raw_fif(raw_fname, add_eeg_ref=False, preload=True,verbose=False)

    #raw.set_eeg_reference()  # set EEG average reference

    baseline = (None, 0)  # means from the first instant to t = 0

    reject = dict(mag=4e-12, eog=150e-6)

    events = mne.read_events(event_fname)

    picks = mne.pick_types(raw.info, meg='mag', eeg=True, eog=True,
                           exclude='bads') #for simplicity ignore grad channels

    raw.rename_channels(mapping={'EOG 061': 'EOG'})

    event_id = {'left/auditory': 1, 'right/auditory': 2,
                'left/visual': 3, 'right/visual': 4}

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=baseline, reject=reject, add_eeg_ref=False, preload=True,verbose=False)
    #print epochs.info
    if treat is not None:
        epochs_eeg = epochs[treat].copy().pick_types(eeg=True,meg=False)
        epochs_meg = epochs[treat].copy().pick_types(meg=True,eeg=False)
    else:
        epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
        epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

    noise_cov = mne.compute_covariance(
        epochs, tmax=0., method=['shrunk', 'empirical'],verbose=False)

    ###############################################################################
    # Inverse modeling: MNE/dSPM on evoked and raw data
    # -------------------------------------------------

    # Read the forward solution and compute the inverse operator

    fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-5-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd, surf_ori=True,verbose=False)

    # Restrict forward solution as necessary
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=True)

    # make an inverse operator
    info = epochs.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                             loose=0.2, depth=0.8,verbose=False)

    write_inverse_operator('sample_audvis-meg-oct-5-inv.fif',
                           inverse_operator,verbose=False)

    ###############################################################################
    # Compute inverse solution
    # ------------------------

    method = "MNE"
    snr = 3.
    lambda2 = 1. / snr ** 2
    if treat is not None:
        stc = apply_inverse_epochs(epochs[treat], inverse_operator, lambda2,
                                   method=method, pick_ori=None,verbose=False)
    else:
        stc = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                   method=method, pick_ori=None,verbose=False)
    if Wt is None:
        print 'Precalculate PCA weights:'
        #weight PCA matrix. Uses 'treat' - so to apply across all treatments, use treat=None
        Wt=Wt_calc(stc,epochs_eeg,epochs_meg,[11,11])
        
    #stc.save('sample_audvis-source-epochs')
 
    if justdims is True:
        meas_dims, m, p, n_steps, total_batch_size, Wt = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
        return meas_dims, m, p, n_steps, total_batch_size, Wt
    else:
        meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
        return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 

def faces_dataset(subject_id,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,treat=None,rnn=True,Wt=None):
    print 'Treat: ', treat
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
    #print epochs.info
    if treat is not None:
        epochs_eeg = epochs[treat].copy().pick_types(eeg=True,meg=False)
        epochs_meg = epochs[treat].copy().pick_types(meg=True,eeg=False)       
    else:
        epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
        epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

    fwd = mne.read_forward_solution(fwdpath,verbose=False)
    inv = read_inverse_operator(invpath,verbose=False)
    method = "MNE"
    snr = 3.
    lambda2 = 1. / snr ** 2
    if treat is not None:
        stc = apply_inverse_epochs(epochs[treat], inv, lambda2,
                                   method=method, pick_ori=None,verbose=False)
    else:
        stc = apply_inverse_epochs(epochs, inv, lambda2,
                                   method=method, pick_ori=None,verbose=False)
        
    if Wt is None:
        print 'Precalculate PCA weights:'
        #weight PCA matrix. Uses 'treat' - so to apply across all treatments, use treat=None
        Wt=Wt_calc(stc,epochs_eeg,epochs_meg,[11,11])
        
    if justdims is True:
        meas_dims, m, p, n_steps, total_batch_size, Wt = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
        return meas_dims, m, p, n_steps, total_batch_size, Wt
    else:
        meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size,Wt = prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
        return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 

        
def prepro(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True,Wt=None):
    if cnn is True:
        if justdims is True:
            meas_dims, m, p, n_steps, total_batch_size, Wt = cnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_dims, m, p, n_steps, total_batch_size, Wt
        else:
            meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt = cnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt
    elif cnn is 'fft':
        if justdims is True:
            meas_dims, m, p, n_steps, total_batch_size, Wt = fftcnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_dims, m, p, n_steps, total_batch_size, Wt
        else:
            meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt = fftcnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt
    else:
        if justdims is True:
            meas_dims, m, p, n_steps, total_batch_size, Wt = xcnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_dims, m, p, n_steps, total_batch_size, Wt            
        else:
            meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt = xcnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection=selection,pca=pca,subsample=subsample,justdims=justdims,cnn=cnn,locate=locate,rnn=rnn,Wt=Wt)
            return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt

def Wt_calc(stc,epochs_eeg,epochs_meg,meas_dims):
    total_batch_size = len(stc)#number of events. we'll consider each event an example.

    n_steps = stc[0]._data.shape[1]
    
    n_eeg = epochs_eeg.get_data().shape[1]

    eeg_xyz=np.squeeze(np.array([epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

    n_meg = epochs_meg.get_data().shape[1]

    meg_xyz=np.squeeze(np.array([epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

    eeg_data = np.array(epochs_eeg.get_data())#batch_sizexmxn_steps

    meg_data = np.array(epochs_meg.get_data())#batch_sizexmxn_steps

    tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)

    Wt=tf_meas.pca()
    
    return Wt

def location(stc,subject,selection='all',locate=True):
    if locate is True: locate=1
    print "Locate ",locate," dipoles"
    if selection is 'all':
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(stc),ns,3*locate))
            
        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(stc)):
            #max location (index)
            mxloca = np.argsort(np.abs(stc[s].data),axis=0)
            mxloc=mxloca[-1-locate:-1,:]
            assert mxloc.shape[0]==locate and mxloc.shape[1]==ns
            hemi = np.where(mxloc<nd/2,0,1).reshape([-1])
            mxvtx_long =vtx_long[mxloc].reshape([-1])
            if subject is 'sample':
                #ns*locatex3
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False)
                
            else:
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False)

            assert tmp.shape[1]==3 and tmp.shape[0]==ns*locate, tmp.shape
            tmp = tmp.reshape([locate,ns,3])
            tmp = np.transpose(tmp,(1,0,2)).reshape([-1,locate*3])
            assert tmp.shape[1]==3*locate and tmp.shape[0]==ns, tmp.shape
            
            loc[s,: ,:] = tmp
            
        qtrue_all = loc
        p=loc.shape[2]
        return qtrue_all, p
    else:        
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(selection),ns,3*locate))

        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        ind_s = 0
        for s in selection:
            #max location (index)
            mxloca = np.argsort(np.abs(stc[s].data),axis=0)
            mxloc=mxloca[-1-locate:-1,:]
            assert mxloc.shape[0]==locate and mxloc.shape[1]==ns
            hemi = np.where(mxloc<nd/2,0,1).reshape([-1])
            mxvtx_long =vtx_long[mxloc].reshape([-1])
            if subject is 'sample':
                #ns*locatex3
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False)
            else:
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False)

            assert tmp.shape[1]==3 and tmp.shape[0]==ns*locate, tmp.shape
            tmp = tmp.reshape([locate,ns,3])
            tmp = np.transpose(tmp,(1,0,2)).reshape([-1,locate*3])
            assert tmp.shape[1]==3*locate and tmp.shape[0]==ns, tmp.shape
            loc[ind_s,: ,:] = tmp

            ind_s+=1
                
        qtrue_all = loc
        p=loc.shape[2]
        return qtrue_all, p

def locationXYZT(stc,subject,selection='all',locate=True):
    if locate is True: locate=1
    print "Locate ",locate," dipoles"
    if selection is 'all':
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(stc),ns,3*locate))
            
        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        for s in range(0,len(stc)):
            #max location (index)
            [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(stc[s].data))),stc[s].data.shape)
            I,J=i[-1-locate:-1],j[-1-locate:-1]#I is neuron index,J is temporal index
            hemi = np.where(I<nd/2,0,1).reshape([-1])
            mxvtx_long =vtx_long[I].reshape([-1])
            if subject is 'sample':
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,locate*3])
            else:
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,locate*3])
            loc[s,-1,:] = tmp
        qtrue_all = loc
        p=loc.shape[2]
        return qtrue_all, p
    else:        
        nd=stc[0].data.shape[0]
        ns=stc[0].data.shape[1]
        loc=np.zeros((len(selection),ns,3*locate))

        vtx = stc[0].vertices
        vtx_long = np.hstack((stc[0].vertices[0],stc[0].vertices[1]))
        hem0 = np.size(vtx[0])
        hem1 = np.size(vtx[1])
        ind_s = 0
        for s in selection:
            #max location (index)
            [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(stc[s].data))),stc[s].data.shape)
            I,J=i[-1-locate:-1],j[-1-locate:-1]#I is neuron index,J is temporal index
            hemi = np.where(I<nd/2,0,1).reshape([-1])
            mxvtx_long =vtx_long[I].reshape([-1])
            if subject is 'sample':
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,locate*3])
            else:
                tmp = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,locate*3])
            loc[ind_s,-1,:]=tmp
            ind_s+=1
        qtrue_all = loc
        p=loc.shape[2]
        return qtrue_all, p

def cnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
    total_batch_size = len(stc)#number of events. we'll consider each event an example.
    if locate is True:
        p=3
    elif locate>0:
        p=3*locate                
    else:
        p = stc[0]._data.shape[0]
    n_steps = stc[0]._data.shape[1]
    meas_dims=[11,11]
    m = meas_dims[0]*meas_dims[1]
    del stc, epochs, epochs_eeg, epochs_meg
    
    return meas_dims, m, p, n_steps, total_batch_size, Wt

def xcnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
    total_batch_size = len(stc)#number of events. we'll consider each event an example.
    if locate is True:
        p=3
    elif locate>0:
        p=3*locate
    else:
        p = stc[0]._data.shape[0]
    n_steps = stc[0]._data.shape[1]

    n_eeg = epochs_eeg.get_data().shape[1]
    n_meg = epochs_meg.get_data().shape[1]

    meas_dims=n_eeg+n_meg
    print "Meas dims in: ", meas_dims
    m = meas_dims
    del stc, epochs, epochs_eeg, epochs_meg
    return meas_dims, m, p, n_steps, total_batch_size,Wt

def cnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
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
        Wt=tf_meas.pca(Wt=Wt)
    elif pca is False:
        tf_meas.scale()
    else:
        pass

    tf_meas.interp()
    tf_meas.reshape()
    #for b in range(0,n_steps*total_batch_size):
    #    tf_meas.plot(b)
    meas_img_all = tf_meas.meas_img
    m = tf_meas.m

    if selection is 'all':
        dipole=np.array([stc[i]._data for i in range(0,len(stc))]).transpose((1,2,0))
    else:
        dipole=np.array([stc[i]._data for i in selection]).transpose((1,2,0)) 

    if locate is not False:
        if rnn is True or cnn is 'fft':
            qtrue_all, p = locationXYZT(stc,subject,selection=selection,locate=locate)
        else:
            qtrue_all, p = location(stc,subject,selection=selection,locate=locate)
    else:
        if rnn is True or cnn is 'fft':
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=subsample)    
            #bxnxp
        else:
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
            #bxnxp
    del stc, epochs, epochs_eeg, epochs_meg
    return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 

def xcnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
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
        Wt=tf_meas.pca(Wt=Wt)
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

    if locate is not False:
        if rnn is True or cnn is 'fft':
            qtrue_all, p = locationXYZT(stc,subject,selection=selection,locate=locate)
        else:
            qtrue_all, p = location(stc,subject,selection=selection,locate=locate)
    else:
        if rnn is True or cnn is 'fft':
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=subsample)    
            #bxnxp
        else:
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
            #bxnxp
    del stc, epochs, epochs_eeg, epochs_meg
    return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 

def fftcnn_justdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
    total_batch_size = len(stc)#number of events. we'll consider each event an example.
    if locate is True:
        p=3
    elif locate>0:
        p=3*locate
    else:
        p = stc[0]._data.shape[0]
    n_steps = stc[0]._data.shape[1]

    n_eeg = epochs_eeg.get_data().shape[1]
    n_meg = epochs_meg.get_data().shape[1]

    meas_dims=[n_steps, n_eeg+n_meg]
    print "Meas dims in: ", meas_dims
    m = meas_dims
    del stc, epochs, epochs_eeg, epochs_meg
    return meas_dims, m, p, n_steps, total_batch_size, Wt

def fftcnn_xjustdims(stc, epochs, epochs_eeg,epochs_meg,subject,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,rnn=True, Wt=None):
    if selection is 'all':
        total_batch_size = len(stc)#number of evens. we'll consider each event an example.
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

    meas_dims=[n_steps, n_eeg+n_meg]
    print "Meas dims in: ", meas_dims
    tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)
    if pca is True:
        Wt=tf_meas.pca(Wt=Wt)
    elif pca is False:
        tf_meas.scale()
    else:
        pass

    tf_meas.stack_reshape()
    ff=np.fft.fft(tf_meas.meas_stack,axis=1)
    #ff=tf_meas.meas_stack
    
    #print ff.shape
    meas_img_all = np.expand_dims(np.abs(ff)*np.abs(ff),-1)
    #meas_img_all = np.expand_dims(ff,-1)


    #print meas_img_all.shape
    m = tf_meas.m

    if selection is 'all':
        dipole=np.array([stc[i]._data for i in range(0,len(stc))]).transpose((1,2,0))
    else:
        dipole=np.array([stc[i]._data for i in selection]).transpose((1,2,0))

    if locate is not False:
        if rnn is True or cnn is 'fft':
            qtrue_all, p = locationXYZT(stc,subject,selection=selection,locate=locate)
        else:
            qtrue_all, p = location(stc,subject,selection=selection,locate=locate)
    else:
        if rnn is True or cnn is 'fft':
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
            #bxnxp
        else:
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=subsample)    
            #bxnxp
    del stc, epochs, epochs_eeg, epochs_meg
    return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size, Wt 

def ttv(total,test_frac,val_frac,batch_frac,rand_test=True):

    a = np.arange(0,total)

    test_size = max(int(test_frac*total),1)
    val_size = max(int(val_frac*(total-test_size)),1)             
    batch_size = max(int(batch_frac*(total-test_size)),1)

    print 't,t,v',test_size,batch_size, val_size
    if rand_test is True:
        test = np.random.choice(a,test_size,replace=False)
    else:
        test = np.arange(0,test_size)
        
    prob_select = np.ones(a.shape)/float(total-test_size)
    prob_select[test]=0.

    if rand_test is True:                     
        val = np.random.choice(a,val_size,replace=False,p=prob_select)
    else:
        val = np.arange(test_size,test_size+val_size)

    prob_select[val]=0.
    prob_select*=float(total-test_size)/float(total-test_size-val_size)
    
    batches = int((total-val_size-test_size)/batch_size)

                    
    assert np.intersect1d(test,val).size is 0

    batch_num=0
    if rand_test is True:
        batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
    else:
        choose = np.arange(batch_size)
        batch=test_size+val_size+batch_num*batch_size+choose

    assert np.intersect1d(batch,test).size is 0
    assert np.intersect1d(batch,val).size is 0

    print "Train batch ", batch_num#, batch
    prob_select*=float(total-test_size-val_size-batch_size*batch_num)/float(total-test_size-val_size-batch_size*(batch_num+1))
    prob_select[batch]=0.
    batch_list = [batch]
    for batch_num in range(1,batches):
        if rand_test is True:
            batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
        else:
            choose = np.arange(batch_size)
            batch=test_size+val_size+batch_num*batch_size+choose
                            
        assert np.intersect1d(batch,test).size is 0
        assert np.intersect1d(batch,val).size is 0

        print "Train batch ", batch_num#, batch
        if batch_num<batches-1:
            prob_select*=float(total-test_size-val_size-batch_size*batch_num)/float(total-test_size-val_size-batch_size*(batch_num+1))
            prob_select[batch]=0.
        batch_list.append(batch)
        
    print "Batches: ", batches, " Batches*batch_size: ", batches*batch_size, " Train set size: ",(total-val_size-test_size)

    return test, val, batch_list, batches
            
def location_rat(locate,batch_size,n_steps,p,dipole, dipole_xyz):
    #print "xyz ", dipole_xyz.shape
    qtrue_all = np.zeros([batch_size,n_steps,p])
    for s in range(0,dipole.shape[2]):
        mxloca = np.argsort(np.abs(dipole[:,:,s]),axis=0)#dipole is pxnxb
        mxloc=mxloca[-1-locate:-1,:]#locatexn
        loc=dipole_xyz[np.ravel(mxloc),:]#locate*nx3
        #print "loc ", loc.shape
        loc=np.transpose(loc.reshape([-1,n_steps,3]),(1,0,2)).reshape([-1,locate*3])#nx3*locate=nxp
        qtrue_all[s,:,:]=loc#m
    return qtrue_all*1000#mm

def location_rat_XYZT(locate,batch_size,n_steps,p,dipole, dipole_xyz):
    #print "xyz ", dipole_xyz.shape
    qtrue_all = np.zeros([batch_size,n_steps,p])
    for s in range(0,dipole.shape[2]):
        #max location (index)
        [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(dipole[:,:,s]))),dipole[:,:,s].shape)
        I,J=i[-1-locate:-1],j[-1-locate:-1]#I is neuron index,J is temporal index
        loc=dipole_xyz[I,:]#locatex3
        #print "loc ", loc.shape
        loc=loc.reshape([-1])#->locate*3
        #print "loc ", loc.shape
        qtrue_all[s,-1,:]=loc#m
    return qtrue_all*1000.#mm

def rat_prepro(n_chan_in,dipole,dipole_xyz,meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, batch_size,subject,selection='all',pca=True,subsample=1,justdims=False,cnn=True,locate=True,rnn=False,Wt=None):

    if locate is True:
        p=3
    elif locate>0:
        p=3*locate                
    else:
        p=dipole.shape[0]
        
    tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, batch_size)
    if pca is True:
        Wt=tf_meas.pca(Wt = Wt)
    elif pca is False:
        tf_meas.scale()
    else:
        pass

    if n_chan_in is 2:
        if cnn is True:    
            print "Image grid dimensions: ", meas_dims
            tf_meas.interp()
            tf_meas.reshape()
            meas_img_all = tf_meas.meas_img
            m =tf_meas.m
            
        else:
            tf_meas.stack_reshape(n_chan_in=n_chan_in)
            meas_img_all = tf_meas.meas_stack
            m =tf_meas.m0+tf_meas.m1
            meas_dims=m
    elif n_chan_in is 1:
        if cnn is True:    
            print "Image grid dimensions: ", meas_dims
            tf_meas.interp()
            tf_meas.reshape()
            meas_img_all = np.expand_dims(tf_meas.meas_img[:,:,:,:,0],axis=-1)
            m =tf_meas.m
        else:
            tf_meas.stack_reshape(n_chan_in=n_chan_in)
            meas_img_all = tf_meas.meas_stack
            m =tf_meas.m0
            meas_dims=m

    if locate is not False:
        if rnn is True or cnn is 'fft':
            qtrue_all=location_rat_XYZT(locate,batch_size,n_steps,p,dipole, dipole_xyz)
        else:
            qtrue_all=location_rat(locate,batch_size,n_steps,p,dipole, dipole_xyz)
    else:
        
        #print p, dipole.shape, "Dipoles (rat_prepro, before scaling)"

        if rnn is True or cnn is 'fft':
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=subsample)    
            #bxnxp
        else:
            #pxn_stepsxbatchsize
            qtrue_all,p=meas_class.scale_dipole(dipole,subsample=subsample)    
            #bxnxp

    #print p, qtrue_all.shape, "Dipoles (rat_prepro, after scaling)"
    assert qtrue_all.shape == (batch_size,n_steps,p), str(qtrue_all.shape)+' '+str((batch_size,n_steps,p))
    if cnn is True:
        assert meas_img_all.shape == (batch_size,n_steps,meas_dims[0],meas_dims[1],1), str(meas_img_all.shape)+' '+ str((batch_size,n_steps,meas_dims[0],meas_dims[1],1))
    else:
        assert meas_img_all.shape == (batch_size,n_steps,m), str(meas_img_all.shape)+' '+str((batch_size,n_steps,m))
        
    
    return meas_img_all, qtrue_all, meas_dims, m, p, n_steps, batch_size, Wt 

    
