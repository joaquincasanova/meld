import os
import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import csv
import Xmeas_class as meas_class
import mne
import sphere
from mne.datasets import sample
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, apply_inverse_epochs,
                              read_inverse_operator)

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

    print "Train batch ", batch_num, batch
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

        print "Train batch ", batch_num, batch
        if batch_num<batches-1:
            prob_select*=float(total-test_size-val_size-batch_size*batch_num)/float(total-test_size-val_size-batch_size*(batch_num+1))
            prob_select[batch]=0.
        batch_list.append(batch)
        
    print "Batches: ", batches, " Batches*batch_size: ", batches*batch_size, " Train set size: ",(total-val_size-test_size)

    return test, val, batch_list, batches
                
def Wt_calc(stc,epochs_eeg,epochs_meg):
    total_batch_size = len(stc)#number of events. we'll consider each event an example.

    n_steps = stc[0]._data.shape[1]
    meas_dims=[11,11]

    n_eeg = epochs_eeg.get_data().shape[1]

    eeg_xyz=np.squeeze(np.array([epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

    n_meg = epochs_meg.get_data().shape[1]

    meg_xyz=np.squeeze(np.array([epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

    eeg_data = np.array(epochs_eeg.get_data())#batch_sizexmxn_steps

    meg_data = np.array(epochs_meg.get_data())#batch_sizexmxn_steps

    tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, meas_dims, n_steps, total_batch_size)

    self.Wt=tf_meas.pca()

    return Wt


class prepro:
    def __init__(self,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,treat=None,rnn=True,self.Wt=None):
        self.selection=selection
        self.pca=pca
        self.subsample=subsample
        self.justdims=justdims
        self.cnn=cnn
        self.locate=locate
        self.treat=treat
        self.rnn=rnn
        self.Wt=Wt
    def aud_dataset(self):
        print 'Treat: ', self.treat
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
        if self.treat is not None:
            self.epochs_eeg = epochs[self.treat].copy().pick_types(eeg=True,meg=False)
            self.epochs_meg = epochs[self.treat].copy().pick_types(meg=True,eeg=False)
        else:
            self.epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
            self.epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

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
        if self.treat is not None:
            self.stc = apply_inverse_epochs(epochs[self.treat], inverse_operator, lambda2,
                                       method=method, pick_ori=None,verbose=False)
        else:
            self.stc = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                       method=method, pick_ori=None,verbose=False)
        if self.Wt is None:
            print 'Precalculate PCA weights:'
            #weight PCA matrix. Uses 'self.treat' - so to apply across all treatments, use treat=None
            self.Wt=Wt_calc(self.stc,self.epochs_eeg,self.epochs_meg)

        #stc.save('sample_audvis-source-epochs')
        self.prepro()
        
    def faces_dataset(self,subject_id):
        print 'Treat: ', self.treat
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

        if self.treat is not None:
            self.epochs_eeg = epochs[self.treat].copy().pick_types(eeg=True,meg=False)
            self.epochs_meg = epochs[self.treat].copy().pick_types(meg=True,eeg=False)       
        else:
            self.epochs_eeg = epochs.copy().pick_types(eeg=True,meg=False)
            self.epochs_meg = epochs.copy().pick_types(meg=True,eeg=False)

        fwd = mne.read_forward_solution(fwdpath,verbose=False)
        inv = read_inverse_operator(invpath,verbose=False)
        method = "MNE"
        snr = 3.
        lambda2 = 1. / snr ** 2
        if self.treat is not None:
            self.stc = apply_inverse_epochs(epochs[self.treat], inv, lambda2,
                                       method=method, pick_ori=None,verbose=False)
        else:
            self.stc = apply_inverse_epochs(epochs, inv, lambda2,
                                       method=method, pick_ori=None,verbose=False)

        if Wt is None:
            print 'Precalculate PCA weights:'
            #weight PCA matrix. Uses 'treat' - so to apply across all treatments, use treat=None
            self.Wt=Wt_calc(self.stc,self.epochs_eeg,self.epochs_meg)

        self.prepro()
        
    def prepro(self):
        if self.cnn is True:
            if self.justdims is True:
                self.cnn_justdims()
            else:
                self.cnn_xjustdims()
        elif self.cnn is 'fft':
            if self.justdims is True:
                self.fftcnn_justdims()
            else:
                self.fftcnn_xjustdims()
        else:
            if self.justdims is True:
                self.xcnn_justdims()            
            else:
                self.xcnn_xjustdims()

    def location(self):
        if self.self.locate is True: self.locate=1
        print "Locate ",self.locate," dipoles"
        if self.selection is 'all':
            nd=self.stc[0].data.shape[0]
            ns=self.stc[0].data.shape[1]
            loc=np.zeros((len(self.stc),ns,3*self.locate))

            vtx = self.stc[0].vertices
            vtx_long = np.hstack((self.stc[0].vertices[0],self.stc[0].vertices[1]))
            hem0 = np.size(vtx[0])
            hem1 = np.size(vtx[1])
            for s in range(0,len(self.stc)):
                #max location (index)
                mxloca = np.argsort(np.abs(self.stc[s].data),axis=0)
                mxloc=mxloca[-1-self.locate:-1,:]
                assert mxloc.shape[0]==self.locate and mxloc.shape[1]==ns
                hemi = np.where(mxloc<nd/2,0,1).reshape([-1])
                mxvtx_long =vtx_long[mxloc].reshape([-1])
                if subject is 'sample':
                    loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,self.locate*3])

            self.qtrue_all = loc
            self.p=loc.shape[2]
        else:        
            nd=self.stc[0].data.shape[0]
            ns=self.stc[0].data.shape[1]
            loc=np.zeros((len(self.selection),ns,3*self.locate))

            vtx = self.stc[0].vertices
            vtx_long = np.hstack((self.stc[0].vertices[0],self.stc[0].vertices[1]))
            hem0 = np.size(vtx[0])
            hem1 = np.size(vtx[1])
            ind_s = 0
            for s in self.selection:
                #max location (index)
                mxloca = np.argsort(np.abs(self.stc[s].data),axis=0)
                mxloc=mxloca[-1-self.locate:-1,:]
                assert mxloc.shape[0]==self.locate and mxloc.shape[1]==ns
                hemi = np.where(mxloc<nd/2,0,1).reshape([-1])
                mxvtx_long =vtx_long[mxloc].reshape([-1])
                if subject is 'sample':
                    loc[ind_s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[ind_s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,self.locate*3])
                ind_s+=1
            self.qtrue_all = loc
            self.p=loc.shape[2]
            
    def locationXYZT(self):
        if self.locate is True: self.locate=1
        print "Locate ",self.locate," dipoles"
        if self.selection is 'all':
            nd=self.stc[0].data.shape[0]
            ns=self.stc[0].data.shape[1]
            loc=np.zeros((len(self.stc),ns,3*self.locate))

            vtx = self.stc[0].vertices
            vtx_long = np.hstack((self.stc[0].vertices[0],self.stc[0].vertices[1]))
            hem0 = np.size(vtx[0])
            hem1 = np.size(vtx[1])
            for s in range(0,len(self.stc)):
                #max location (index)
                [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(self.stc[s].data))),self.stc[s].data.shape)
                I,J=i[-1-self.locate:-1],j[-1-self.locate:-1]#I is neuron index,J is temporal index
                hemi = np.where(I<nd/2,0,1).reshape([-1])
                mxvtx_long =vtx_long[I].reshape([-1])
                if subject is 'sample':
                    loc[s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,self.locate*3])

            self.qtrue_all = loc
            self.p=loc.shape[2]
        else:        
            nd=self.stc[0].data.shape[0]
            ns=self.stc[0].data.shape[1]
            loc=np.zeros((len(self.selection),ns,3*self.locate))

            vtx = self.stc[0].vertices
            vtx_long = np.hstack((self.stc[0].vertices[0],self.stc[0].vertices[1]))
            hem0 = np.size(vtx[0])
            hem1 = np.size(vtx[1])
            ind_s = 0
            for s in self.selection:
                #max location (index)
                [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(self.stc[s].data))),self.stc[s].data.shape)
                I,J=i[-1-self.locate:-1],j[-1-self.locate:-1]#I is neuron index,J is temporal index
                hemi = np.where(I<nd/2,0,1).reshape([-1])
                mxvtx_long =vtx_long[I].reshape([-1])
                if subject is 'sample':
                    loc[ind_s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[ind_s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,subject,verbose=False).reshape([-1,self.locate*3])
                ind_s+=1
            self.qtrue_all = loc
            self.p=loc.shape[2]

    def cnn_justdims(self):
        self.total_batch_size = len(self.stc)#number of events. we'll consider each event an example.
        if self.locate is True:
            self.p=3
        elif self.locate>0:
            self.p=3*self.locate                
        else:
            p = self.stc[0]._data.shape[0]
        self.n_steps = self.stc[0]._data.shape[1]
        self.meas_dims=[11,11]
        m = self.meas_dims[0]*self.meas_dims[1]
        #del self.stc, epochs, epochs_eeg, epochs_meg

    def xcnn_justdims(self):
        self.total_batch_size = len(self.stc)#number of events. we'll consider each event an example.
        if self.locate is True:
            self.p=3
        elif self.locate>0:
            self.p=3*self.locate
        else:
            p = self.stc[0]._data.shape[0]
        self.n_steps = self.stc[0]._data.shape[1]

        n_eeg = self.epochs_eeg.get_data().shape[1]
        n_meg = self.epochs_meg.get_data().shape[1]

        self.meas_dims=n_eeg+n_meg
        print "Meas dims in: ", self.meas_dims
        self.m =self.meas_dims
        ##del self.stc, epochs, self.epochs_eeg, self.epochs_meg

    def cnn_xjustdims(self):
        if self.selection is 'all':
            self.total_batch_size = len(self.stc)#number of events. we'll consider each event an example.
        else:                
            self.total_batch_size = len(self.selection)#number of events. we'll consider each event an example.

        n_eeg = self.epochs_eeg.get_data().shape[1]

        eeg_xyz=np.squeeze(np.array([self.epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

        n_meg = self.epochs_meg.get_data().shape[1]

        meg_xyz=np.squeeze(np.array([self.epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

        if self.selection is 'all':
            eeg_data = np.array(self.epochs_eeg.get_data())#batch_sizexmxn_steps
            meg_data = np.array(self.epochs_meg.get_data())#batch_sizexmxn_steps
        else:
            eeg_data = np.array(self.epochs_eeg.get_data())[self.selection,:,:]#batch_sizexmxn_steps
            meg_data = np.array(self.epochs_meg.get_data())[self.selection,:,:]#batch_sizexmxn_steps

        self.n_steps=meg_data.shape[2]

        self.meas_dims=[11,11]
        print "Image grid dimensions: ", self.meas_dims
        tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, self.meas_dims, self.n_steps, self.total_batch_size)
        if self.pca is True:
            self.Wt=tf_meas.pca(Wt=self.Wt)
        elif self.pca is False:
            tf_meas.scale()
        else:
            pass

        tf_meas.interp()
        tf_meas.reshape()
        #for b in range(0,n_steps*total_batch_size):
        #    tf_meas.plot(b)
        meas_img_all = tf_meas.meas_img
        self.m =tf_meas.m

        if self.selection is 'all':
            dipole=np.array([self.stc[i]._data for i in range(0,len(self.stc))]).transpose((1,2,0))
        else:
            dipole=np.array([self.stc[i]._data for i in self.selection]).transpose((1,2,0)) 

        if self.locate is not False:
            if self.rnn is True or self.cnn is 'fft':
                self.locationXYZT()
            else:
                self.location()
        else:
            if self.rnn is True or self.cnn is 'fft':
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
        #del self.stc, epochs, self.epochs_eeg, self.epochs_meg
        
    def xcnn_xjustdims(self):
        if self.selection is 'all':
            self.total_batch_size = len(self.stc)#number of events. we'll consider each event an example.
        else:
            self.total_batch_size = len(self.selection)#number of events. we'll consider each event an example.

        n_eeg = self.epochs_eeg.get_data().shape[1]

        eeg_xyz=np.squeeze(np.array([self.epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

        n_meg = self.epochs_meg.get_data().shape[1]

        meg_xyz=np.squeeze(np.array([self.epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

        if self.selection is 'all':
            eeg_data = np.array(self.epochs_eeg.get_data())#batch_sizexmxn_steps
            meg_data = np.array(self.epochs_meg.get_data())#batch_sizexmxn_steps
        else:
            eeg_data = np.array(self.epochs_eeg.get_data())[self.selection,:,:]#batch_sizexmxn_steps
            meg_data = np.array(self.epochs_meg.get_data())[self.selection,:,:]#batch_sizexmxn_steps

        self.n_steps=meg_data.shape[2]

        self.meas_dims=n_eeg+n_meg
        print "Meas dims in: ", self.meas_dims
        tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, self.meas_dims, self.n_steps, self.total_batch_size)
        if self.pca is True:
            self.Wt=tf_meas.pca(Wt=self.Wt)
        elif self.pca is False:
            tf_meas.scale()
        else:
            pass

        tf_meas.stack_reshape()
        meas_img_all = tf_meas.meas_stack
        self.m =tf_meas.m

        if self.selection is 'all':
            dipole=np.array([self.stc[i]._data for i in range(0,len(self.stc))]).transpose((1,2,0))
        else:
            dipole=np.array([self.stc[i]._data for i in self.selection]).transpose((1,2,0))

        if self.locate is not False:
            if self.rnn is True or self.cnn is 'fft':
                self.locationXYZT()
            else:
                self.location()
        else:
            if self.rnn is True or self.cnn is 'fft':
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
        #del self.stc, epochs, self.epochs_eeg, self.epochs_meg

    def fftcnn_justdims(self):
        self.total_batch_size = len(self.stc)#number of events. we'll consider each event an example.
        if self.locate is True:
            self.p=3
        elif self.locate>0:
            self.p=3*self.locate
        else:
            p = self.stc[0]._data.shape[0]
        self.n_steps = self.stc[0]._data.shape[1]

        n_eeg = self.epochs_eeg.get_data().shape[1]
        n_meg = self.epochs_meg.get_data().shape[1]

        self.meas_dims=[self.n_steps, n_eeg+n_meg]
        print "Meas dims in: ", self.meas_dims
        self.m =self.meas_dims
        #del self.stc, epochs, self.epochs_eeg, self.epochs_meg

    def fftcnn_xjustdims(self):
        if self.selection is 'all':
            self.total_batch_size = len(self.stc)#number of evens. we'll consider each event an example.
        else:
            self.total_batch_size = len(self.selection)#number of events. we'll consider each event an example.

        n_eeg = self.epochs_eeg.get_data().shape[1]

        eeg_xyz=np.squeeze(np.array([self.epochs_eeg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_eeg)]))

        n_meg = self.epochs_meg.get_data().shape[1]

        meg_xyz=np.squeeze(np.array([self.epochs_meg.info['chs'][i]['loc'][:3].reshape([1,3]) for i in range(0,n_meg)]))

        if self.selection is 'all':
            eeg_data = np.array(self.epochs_eeg.get_data())#batch_sizexmxself.n_steps
            meg_data = np.array(self.epochs_meg.get_data())#batch_sizexmxn_steps
        else:
            eeg_data = np.array(self.epochs_eeg.get_data())[self.selection,:,:]#batch_sizexmxn_steps
            meg_data = np.array(self.epochs_meg.get_data())[self.selection,:,:]#batch_sizexmxn_steps

        self.n_steps=meg_data.shape[2]

        self.meas_dims=[self.n_steps, n_eeg+n_meg]
        print "Meas dims in: ", self.meas_dims
        tf_meas = meas_class.meas(meg_data,meg_xyz, eeg_data,eeg_xyz, self.meas_dims, self.n_steps, self.total_batch_size)
        if self.pca is True:
            self.Wt=tf_meas.pca(Wt=self.Wt)
        elif self.pca is False:
            tf_meas.scale()
        else:
            pass

        tf_meas.stack_reshape()
        #ff=np.fft.fft(tf_meas.meas_stack,axis=1)
        ff=tf_meas.meas_stack

        #print ff.shape
        #meas_img_all = np.expand_dims(np.abs(ff)*np.abs(ff),-1)
        meas_img_all = np.expand_dims(ff,-1)


        #print meas_img_all.shape
        self.m =tf_meas.m

        if self.selection is 'all':
            dipole=np.array([self.stc[i]._data for i in range(0,len(self.stc))]).transpose((1,2,0))
        else:
            dipole=np.array([self.stc[i]._data for i in self.selection]).transpose((1,2,0))

        if self.locate is not False:
            if self.rnn is True or self.cnn is 'fft':
                self.locationXYZT()
            else:
                self.location()
        else:
            if self.rnn is True or self.cnn is 'fft':
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
        #del self.stc, epochs, self.epochs_eeg, self.epochs_meg

        
