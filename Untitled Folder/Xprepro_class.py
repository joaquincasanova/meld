import os
import numpy as np
from numpy import matlib

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sphere
import dipole_class_rat
import csv
import meas_class
import mne
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

    Wt=tf_meas.pca()

    return Wt


class prepro:
    def __init__(self,selection='all',pca=False,subsample=1,justdims=True,cnn=True,locate=True,treat=None,rnn=True,Wt=None):
        self.selection=selection
        self.pca=pca
        self.subsample=subsample
        self.justdims=justdims
        self.cnn=cnn
        self.locate=locate
        self.treat=treat
        self.rnn=rnn
        self.Wt=Wt

    def rat_synth(self,total_batch_size,delT,n_steps,meas_dims,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True):
        self.total_batch_size=total_batch_size
        if self.selection is 'all':
            self.batch_size=self.total_batch_size
        else:
            self.batch_size=len(self.selection)
        self.subject='rat'
        self.delT=delT
        self.n_steps=n_steps
        self.meas_dims=meas_dims
        self.meas_xyz=meas_xyz
        self.dipole_xyz=dipole_xyz
        self.dipole_dims=dipole_dims
        self.n_chan_in=n_chan_in
        self.noise_flag=noise_flag
        self.orient=orient

        instance = dipole_class_rat.dipole(self.delT, self.batch_size, self.n_steps,
                          2, self.meas_dims, self.dipole_dims,
                          orient=self.orient, noise_flag=self.noise_flag,
                          dipole_xyz=self.dipole_xyz, meas_xyz=self.meas_xyz,pca=self.pca)
        instance.batch_sequence_gen()
        
        self.meg_data=instance.meg_data
        self.eeg_data=instance.eeg_data
        self.dipole=instance.qtrue
        self.m=instance.m
        self.p=instance.p
        self.meg_xyz=instance.meg_xyz
        assert self.meg_xyz.shape[0]==self.m and self.meg_xyz.shape[1]==3, self.meg_xyz.shape
        self.eeg_xyz=instance.eeg_xyz
        assert self.eeg_xyz.shape[0]==self.m and self.eeg_xyz.shape[1]==3, self.eeg_xyz.shape
        
        self.dipole_xyz=instance.dipole_xyz
        assert self.dipole_xyz.shape[0]==self.p and self.dipole_xyz.shape[1]==3, self.dipole_xyz.shape
        self.n_steps=instance.n_steps
        self.batch_size=instance.batch_size
        
        if self.locate is True:
            self.p=3
        elif self.locate>0:
            self.p=3*self.locate                
        else:
            self.p=instance.p
        self.rat_prepro()
        
    def aud_dataset(self):
        print 'Treat: ', self.treat
        ###############################################################################
        # Setup for reading the raw data
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
        event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
        tmin = -0.2  # start of each epoch (200ms before the trigger)
        tmax = 0.5  # end of each epoch (500ms after the trigger)
        self.subject='sample'
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

        self.subject = "sub%03d" % subject_id
        print("processing %s" % self.subject)

        invname = '%s-meg-%s-inv.fif' % (self.subject,spacing)
        invpath = os.path.join(os.path.join(meg_dir,self.subject),invname) 
        fwdname = '%s-meg-%s-fwd.fif' % (self.subject,spacing)
        fwdpath = os.path.join(os.path.join(meg_dir,self.subject),fwdname) 
        eponame = '%s-epo.fif' % (self.subject) 
        epopath = os.path.join(os.path.join(meg_dir,self.subject),eponame)
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

        if self.Wt is None:
            print 'Precalculate PCA weights:'
            #weight PCA matrix. Uses 'treat' - so to apply across all treatments, use treat=None
            self.Wt=Wt_calc(self.stc,self.epochs_eeg,self.epochs_meg)

        self.prepro()

    def dipole_hist(self):
        if self.treat is None:
            lab='All'
        else:
            lab=self.treat
        nd=self.stc[0].data.shape[0]
        ns=self.stc[0].data.shape[1]
        if self.selection is 'all':
            x = np.vstack((np.abs(self.stc[0].data),np.abs(self.stc[1].data)))
            for s in range(2,len(self.stc)):
                x = np.vstack((x,np.abs(self.stc[s].data)))
        else:
            x = np.vstack((np.abs(self.stc[self.selection[0]].data),np.abs(self.stc[self.selection[1]].data)))
            for s in self.selection[2:]:
                x = np.vstack((x,np.abs(self.stc[s].data)))
            
            # the histogram of the data
        #for t in range(0,ns):
        n, bins, patches = plt.hist(np.log(x.reshape([-1,1])), 500, normed=1, facecolor='green', alpha=0.75)
            
        plt.xlabel('Current')
        plt.ylabel('Probability')
        plt.title('Histogram of Current: '+lab)
        #plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        
        plt.show()
            
    def location_rat(self):
        self.qtrue_all = np.zeros([self.batch_size,self.n_steps,self.p])
        for s in range(0,self.dipole.shape[2]):
            mxloca = np.argsort(np.abs(self.dipole[:,:,s]),axis=0)#dipole is pxnxb
            mxloc=mxloca[-1-self.locate:-1,:]#locatexn
            loc=self.dipole_xyz[np.ravel(mxloc),:]#locate*nx3
            loc=np.transpose(loc.reshape([-1,self.n_steps,3]),(1,0,2)).reshape([-1,self.locate*3])#nx3*locate=nxp
            self.qtrue_all[s,:,:]=loc*1000.#mm
        
    def location_rat_XYZT(self):
        self.qtrue_all = np.zeros([self.batch_size,self.n_steps,self.p])
        for s in range(0,self.dipole.shape[2]):
            #max location (index)
            [i,j] = np.unravel_index(np.argsort(np.ravel(np.abs(self.dipole[:,:,s]))),self.dipole[:,:,s].shape)
            I,J=i[-1-self.locate:-1],j[-1-self.locate:-1]#I is neuron index,J is temporal index
            loc=self.dipole_xyz[I,:]#locatex3
            loc=loc.reshape([-1])#->locate*3
            self.qtrue_all[s,-1,:]=loc*1000.#mm
            
    def rat_prepro(self):
        tf_meas = meas_class.meas(self.meg_data,self.meg_xyz, self.eeg_data,self.eeg_xyz, self.meas_dims, self.n_steps, self.batch_size)
        if self.pca is True:
            self.Wt=tf_meas.pca(Wt = self.Wt)
        elif self.pca is False:
            tf_meas.scale()
        else:
            pass
        
        if self.n_chan_in is 2:
            if self.cnn is True:    
                print "Image grid dimensions: ", self.meas_dims
                tf_meas.interp()
                tf_meas.reshape()
                self.meas_img_all = tf_meas.meas_img
                self.m =tf_meas.m
            else:
                self.meas_img_all = np.vstack((tf_meas.meas_in[0],tf_meas.meas_in[1]))
                self.m =tf_meas.m0+tf_meas.m1
        elif self.n_chan_in is 1:
            if self.cnn is True:    
                print "Image grid dimensions: ", self.meas_dims
                tf_meas.interp()
                tf_meas.reshape()
                self.meas_img_all = np.expand_dims(tf_meas.meas_img[:,:,:,:,0],axis=-1)
                self.m =tf_meas.m
            else:
                self.meas_img_all = np.vstack((tf_meas.meas_in[0]))
                self.m =tf_meas.m0
            
        if self.locate is not False:
            if self.rnn is True or self.cnn is 'fft':
                self.location_rat_XYZT()
            else:
                self.location_rat()
        else:
            if self.rnn is True or self.cnn is 'fft':
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(self.dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(self.dipole,subsample=self.subsample)    
                #bxnxp
        
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
                mxloca = np.argsort(np.abs(self.stc[s].data),axis=0)
                mxloc=mxloca[-1-self.locate:-1,:]
                assert mxloc.shape[0]==self.locate and mxloc.shape[1]==ns
                hemi = np.where(mxloc<nd/2,0,1).reshape([-1])
                mxvtx_long =vtx_long[mxloc].reshape([-1])
                if self.subject is 'sample':
                    loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,verbose=False).reshape([-1,self.locate*3])

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
                if self.subject is 'sample':
                    loc[ind_s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[ind_s,: ,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,verbose=False).reshape([-1,self.locate*3])
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
                if self.subject is 'sample':
                    loc[s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,verbose=False).reshape([-1,self.locate*3])

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
                if self.subject is 'sample':
                    loc[ind_s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,subjects_dir='/home/jcasa/mne_data/MNE-sample-data/subjects',verbose=False).reshape([-1,self.locate*3])
                else:
                    loc[ind_s,-1,:] = mne.vertex_to_mni(mxvtx_long,hemi,self.subject,verbose=False).reshape([-1,self.locate*3])
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
        self.meas_img_all = tf_meas.meas_img
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
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
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
        self.meas_img_all = tf_meas.meas_stack
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
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
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
        self.meas_img_all = np.expand_dims(ff,-1)


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
                self.qtrue_all,self.p=meas_class.scale_dipoleXYZT_OH(dipole,subsample=self.subsample)    
                #bxnxp
            else:
                #pxn_stepsxbatchsize
                self.qtrue_all,self.p=meas_class.scale_dipole(dipole,subsample=self.subsample)    
                #bxnxp
        #del self.stc, epochs, self.epochs_eeg, self.epochs_meg

