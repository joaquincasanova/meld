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

def pred_obs_last(guess,true,locate,name):
    
    if locate is 1 or locate is True:     
        z = np.squeeze(guess[:,2])
        y = np.squeeze(guess[:,1])
        x = np.squeeze(guess[:,0])

        zt = np.squeeze(true[:,2])
        yt = np.squeeze(true[:,1])
        xt = np.squeeze(true[:,0])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, 'ob')
        ax.plot(xt, yt, zt, 'xr')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Dipole locations (mm)')
        plt.savefig(name+'.png')
        plt.close()
        ############################################################################
        ###

def pred_obs_series(guess,true,locate,name,n,b):
    guess=guess.reshape([b,n,-1])
    true=true.reshape([b,n,-1])
    for t in range(0,n):
        if locate is 1 or locate is True:     
            z = np.squeeze(guess[:,t,2])
            y = np.squeeze(guess[:,t,1])
            x = np.squeeze(guess[:,t,0])

            zt = np.squeeze(true[:,t,2])
            yt = np.squeeze(true[:,t,1])
            xt = np.squeeze(true[:,t,0])

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(x, y, z, 'ob')
            ax.plot(xt, yt, zt, 'xr')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title('Dipole locations (mm) at timestep #'+str(t))
            plt.savefig(name+'_'+str(t)'.png')
            plt.close()
    ############################################################################
    ###

    
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

for cv_run in [0, 1, 2, 3, 4]:
    for locate in [1,100]:
        for cnn in [True,False]:
            for rnn in [True,False]:
                for subject_id in ['aud',7,8]:
                    if subject_id is 'aud':
                        treats=[None,'left/auditory', 'right/auditory', 'left/visual', 'right/visual']
                    elif subject_id is 7 or subject_id is 8:
                        treats=[None,'face/famous','scrambled','face/unfamiliar']

                    for treat in treats:
                        if treat is not None:
                            lab_treat=treat.replace("/","_")
                        else:
                            lab_treat='None'

                        print 'Subject: ',subject_id,' PCA: ',pca,' Random: ',rand_test, ' CNN: ',cnn, ' RNN: ',rnn, 'Locate: ',locate, 'Treat: ',lab_treat

                        fieldnames=['batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','cost']
                        name='/home/jcasa/meld/tnrse2017/data/tf1_%s_subject_%s_pca_all_%s_rand_%s_cnn_%s_rnn_%s_locate_%s_treat_%s' % (cv_run,subject_id, pca, rand_test, cnn, rnn,locate,lab_treat)
                        fname = name + '.csv' 

                        with open(fname,'w') as csvfile:
                            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                            writer.writeheader()

                            for [k_conv, n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:

                                n_chan_in=2

                                if subject_id is 'aud':
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=None)
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                else:
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=None)
                                    meas_dims, m, p, n_steps, total_batch_size,Wt = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate,treat=treat,Wt=Wt)

                                test, val, batch_list, batches = nn_prepro.ttv(total_batch_size,test_frac,val_frac,batch_frac,rand_test=rand_test)

                                per_batch = int(5000/batches)
                                if subject_id is 'aud':
                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size,Wt = nn_prepro.aud_dataset(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                else:
                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size,Wt = nn_prepro.faces_dataset(subject_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
                                #pick a test batch
                                print "Test batch "#,test

                                if subject_id is 'aud':
                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size,Wt = nn_prepro.aud_dataset(selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,Wt=Wt)
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

                                            if step % plot_step==0:
                                                if rnn is True:
                                                    guess,true = session.run([nn.qhat_last, nn.qtrain_last],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                    pred_obs_last(guess,true,locate,name+'_train_'+str(step))
                                                else:
                                                    guess,true = session.run([nn.qhat, nn.qtrain_unflat],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                                    pred_obs_series(guess,true,locate,name+'_train_'+str(step),n_steps,batch_size)
                                                
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
                                        if rnn:
                                            costt, guess, true = session.run([nn.cost, nn.qhat_last, nn.qtrain_last],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                            pred_obs_last(guess,true,locate,name)

                                        else:
                                            costt, guess, true = session.run([nn.cost, nn.qhat, nn.qtrain_unflat],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                            pred_obs_series(guess,true,locate,name,n_steps,test_size)

                                        print "Test Step: ", step, "Cost: ", costt
                                        
                                    writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'cost':costt})
                                    


                        csvfile.close()
