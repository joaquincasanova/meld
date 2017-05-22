import numpy as np
from numpy import matlib
import sphere
import tensorflow as tf
import meld_net_1 as meld_net
import csv
import nn_prepro
import time
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def pred_obs(guess,true,locate,name):
    for l in range(0,locate):
        #guess=sphere.sph2cartMatNB(guess)
        #true=sphere.sph2cartMatNB(true)
        
        z = np.squeeze(guess[:,2])
        y = np.squeeze(guess[:,1])
        x = np.squeeze(guess[:,0])

        zt = np.squeeze(true[:,2+l*3])
        yt = np.squeeze(true[:,1+l*3])
        xt = np.squeeze(true[:,0+l*3])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, 'ob')
        ax.plot(xt, yt, zt, 'xr')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Dipole locations (mm)')
        plt.savefig(name+'.png')
        plt.close()
    ###############################################################################

#meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True, subsample=10)

pca = True
rand_test = True
plot_step = 500
val_step = 100
print_step = 10
learning_rate = 0.005
dropout = 1.
beta = 0.
subsample = 1
params_list = [[3,3,5,10,3,.2,.2,.2]]

for cv_run in [3,4]:
    for locate in [10]:
        for cnn in [True,False]:
            for rnn in [True,False]:
                for subject_id in ['aud',7]:
                    if subject_id is 'aud':
                        treats=[None]#,'left/auditory', 'right/auditory', 'left/visual', 'right/visual']
                    elif subject_id is 'rat':
                        treats=[None]
                    else:
                        treats=[None]#,'face/famous','scrambled','face/unfamiliar']

                    for treat in treats:
                        if treat is not None:
                            lab_treat=treat.replace("/","_")
                        else:
                            lab_treat='None'

                        print 'Subject: ',subject_id,' PCA: ',pca,' Random: ',rand_test, ' CNN: ',cnn, ' RNN: ',rnn, 'Locate: ',locate, 'Treat: ',lab_treat

                        fieldnames=['batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','cost']
                        name='/home/jcasa/meld/tnrse2017/tf1_%s_subject_%s_pca_all_%s_rand_%s_cnn_%s_rnn_%s_locate_%s_treat_%s' % (cv_run,subject_id, pca, rand_test, cnn, rnn,locate,lab_treat)
                        fname = name + '.csv' 

                        with open(fname,'w') as csvfile:
                            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                            writer.writeheader()

                            for [k_conv, n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:

                                if cnn is 'fft' or subject_id is 'rat':
                                    n_chan_in=1
                                else:
                                    n_chan_in=2

                                if subject_id is 'aud':
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=None)
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                elif subject_id is 'rat':
                                    total_batch_size=1000
                                    delT=1e-2
                                    n_steps=100
                                    meas_dims_in=[4,1]
                                    dipole_dims=[1,1,4]
                                    if cnn is True:
                                        assert k_conv<np.min(meas_dims), "Kconv must be less than image size."
                                    meas_dims, m, p, n_steps, total_batch_size, Wt = nn_prepro.rat_synth(total_batch_size,delT,n_steps,meas_dims_in,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection='all',pca=True,subsample=1,justdims=True,cnn=cnn,locate=locate,treat=None,rnn=rnn,Wt=None)
                                    meas_dims, m, p, n_steps, total_batch_size, Wt = nn_prepro.rat_synth(total_batch_size,delT,n_steps,meas_dims_in,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection='all',pca=True,subsample=1,justdims=True,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                                    #print p, "Dipoles returned"
                                else:
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=None)
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=treat,Wt=Wt)

                                test, val, batch_list, batches = nn_prepro.ttv(total_batch_size,test_frac,val_frac,batch_frac,rand_test=rand_test)

                                per_batch = int(5000/batches)
                                if subject_id is 'aud':
                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size,Wt = nn_prepro.aud_dataset(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                elif subject_id is 'rat':
                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size,Wt = nn_prepro.rat_synth(total_batch_size,delT,n_steps,meas_dims_in,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                                    #print p, "Dipoles returned"
                                else:
                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size,Wt = nn_prepro.faces_dataset(subject_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                #pick a test batch
                                print "Test batch "#,test

                                if subject_id is 'aud':
                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size,Wt = nn_prepro.aud_dataset(selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                elif subject_id is 'rat':
                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size,Wt = nn_prepro.rat_synth(total_batch_size,delT,n_steps,meas_dims_in,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                                    #print p, "Dipoles returned"
                                else:
                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size,Wt = nn_prepro.faces_dataset(subject_id,selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                #pick a val batch
                                print "Val batch "#,val

                                n_out=p
                                k_pool=1

                                print "Meas: ", m, " Out: ",p, " Steps: ",n_steps
                                time.sleep(10)
                                nn=meld_net.meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cnn=cnn,rnn=rnn,locate=locate)
                                tf.reset_default_graph()
                                nn.network()
                                nn.cost()
                                nn.trainer()
                                nn.initializer()     

                                with tf.Session() as session:
                                    logdir = '/tmp/tensorflowlogs/tnrse2017_tf1_%s/sub_%s/pca_all_%s/rand_%s/cnn_%s/rnn_%s/locate_knn_%s/treat_%s/' % (cv_run,subject_id,pca,rand_test,cnn,rnn,locate,lab_treat)
                                    if tf.gfile.Exists(logdir):
                                        tf.gfile.DeleteRecursively(logdir)
                                    tf.gfile.MakeDirs(logdir)
                                    train_writer = tf.summary.FileWriter(logdir,session.graph)

                                    session.run(nn.init_step)

                                    for batch_num in range(0,batches):
                                        err_l_prev = 1000.
                                        err_l = 500.
                                        batch = batch_list[batch_num]
                                        print "Train batch ", batch_num#, batch
                                        #pick a first batch of batch_size
                                        if subject_id is 'aud':
                                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size,Wt = nn_prepro.aud_dataset(selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                        elif subject_id is 'rat':
                                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size,Wt = nn_prepro.rat_synth(total_batch_size,delT,n_steps,meas_dims_in,dipole_dims,n_chan_in,meas_xyz=None,dipole_xyz=None,orient=None,noise_flag=True,selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                                        else:
                                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size,Wt = nn_prepro.faces_dataset(subject_id,selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)

                                        step=0
                                        while step<per_batch:
                                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                            run_metadata = tf.RunMetadata()

                                            if locate is False:
                                                train_summary,train_acc_summary, _ ,cost,acc = session.run([nn.train_summary,nn.train_acc_summary, nn.train_step, nn.cost, nn.accuracy],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                            else:
                                                train_summary, _ ,cost = session.run([nn.train_summary, nn.train_step, nn.cost],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})

                                            if step % print_step==0:
                                                print "Train Step: ", step, "Cost: ",cost                                              

                                            if False:#step % plot_step==0:
                                                name = str(subject_id)+'_'+str(batch_num)+'_'+str(step)+'_'+str(rnn)+'_'+str(cnn)
                                                if rnn is True:
                                                    guess,true = session.run([nn.qhat_last, nn.qtrain_last],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                else:
                                                    guess,true = session.run([nn.qhat, nn.qtrain_unflat],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                pred_obs(guess,true,locate,name)

                                            writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'cost':cost})

                                            tstep=step+batch_num*per_batch
                                            train_writer.add_run_metadata(run_metadata, 'train_step%03d' % tstep)
                                            if locate is False:
                                                train_writer.add_summary(train_summary, tstep)
                                                train_writer.add_summary(train_acc_summary, tstep)
                                            else:
                                                train_writer.add_summary(train_summary, tstep)

                                            if step % val_step==0 and step!=0:
                                                if locate is False:
                                                    valid_summary,valid_acc_summary,costv,accv = session.run([nn.valid_summary,nn.valid_acc_summary, nn.cost, nn.accuracy], feed_dict={nn.qtrainPH: qtrue_val, nn.measPH: meas_img_val, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                    train_writer.add_summary(valid_summary, tstep)
                                                    train_writer.add_summary(valid_acc_summary, tstep)
                                                else:
                                                    valid_summary,costv = session.run([nn.valid_summary, nn.cost], feed_dict={nn.qtrainPH: qtrue_val, nn.measPH: meas_img_val, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                    train_writer.add_summary(valid_summary, tstep)
                                                print "Val Step: ", step, "Cost: ",costv

                                                writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'cost':costv})

                                            step+=1


                                    #save_path = nn.saver.save(session, "./data/model.ckpt")
                                    #print("Model saved in file: %s" % save_path)

                                    #test batch

                                    if locate is False:
                                        costt, acct = session.run([nn.cost, nn.accuracy],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                    else:
                                        costt = session.run([nn.cost],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                    print "Test Step: ", step, "Cost: ", costt
                                    writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'cost':costt})




                        csvfile.close()
