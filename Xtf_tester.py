import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import Xmeld_net as meld_net
import csv
import prepro_class
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
        ax.plot(xt, yt, zt, 'or')
        plt.title('Number '+str(l))
        plt.savefig(name+'.png')
        plt.close()
    ###############################################################################

#meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True, subsample=10)

pca = True
rand_test = True
plot_step = 1000
val_step = 100
print_step = 10
learning_rate = 0.005
dropout = 1.
beta = 0.

for locate in [95,10,1]:
    subsample = 1
    if locate  is False:
        subsample=20
    for cnn in [True,False]:
        if cnn is 'fft':
            params_list = [[25,2,3,100,3,.2,.2,.2]]
        else:
            params_list = [[3,3,5,10,3,.2,.2,.2]]

        for rnn in [True,False]:
            for subject_id in ['aud']:
                if subject_id is 'aud':
                    treats=['left/auditory', 'right/auditory', 'left/visual', 'right/visual',None]
                else:
                    treats=['face/famous','scrambled','face/unfamiliar']
                treats = [None]
                prepro = prepro_class.prepro(selection='all',pca=pca,subsample=subsample,justdims=True,cnn=cnn,locate=locate,treat=None,rnn=rnn,Wt=None)
                if subject_id is 'aud':
                    prepro.aud_dataset()
                else:
                    prepro.faces_dataset(subject_id)

                Wt = prepro.Wt#calculate with ALL
                
                for treat in treats:

                    prepro = prepro_class.prepro(selection='all',pca=pca,subsample=subsample,justdims=True,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                    if subject_id is 'aud':
                        prepro.aud_dataset()
                    else:
                        prepro.faces_dataset(subject_id)        
                    
                    if treat is not None:
                        lab_treat=treat.replace("/","_")
                    else:
                        lab_treat='None'
                        
                    print 'Subject: ',subject_id,' PCA: ',pca,' Random: ',rand_test, ' CNN: ',cnn, ' RNN: ',rnn, 'Locate: ',locate, 'Treat: ',lab_treat

                    fieldnames=['batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','cost']
                    name='./data/X_subject_%s_pca_all_%s_rand_%s_cnn_%s_rnn_%s_locate_%s_treat_%s' % (subject_id, pca, rand_test, cnn, rnn,locate,lab_treat)
                    fname = name + '.csv' 

                    with open(fname,'a') as csvfile:
                        writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                        writer.writeheader()

                        for [k_conv, n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:

                            test, val, batch_list, batches = prepro_class.ttv(prepro.total_batch_size,test_frac,val_frac,batch_frac,rand_test=rand_test)
                          
                            if cnn is 'fft':
                                n_chan_in=1
                            else:
                                n_chan_in=2


                            prepro = prepro_class.prepro(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                            if subject_id is 'aud':
                                prepro.aud_dataset()
                            else:
                                prepro.faces_dataset(subject_id)
                            meas_img_test = prepro.meas_img_all
                            qtrue_test = prepro.qtrue_all

                            prepro = prepro_class.prepro(selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                            if subject_id is 'aud':
                                prepro.aud_dataset()
                            else:
                                prepro.faces_dataset(subject_id)
                            meas_img_val = prepro.meas_img_all
                            qtrue_val = prepro.qtrue_all
                            p=prepro.p
                            m=prepro.m
                            n_steps=prepro.n_steps
                            meas_dims=prepro.meas_dims
                            
                            per_batch = int(5000/batches)
                            
                            n_out=p
                            k_pool=1

                            print "Meas: ", m, " Out: ",p, " Steps: ",n_steps

                            nn=meld_net.meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cnn=cnn,rnn=rnn,locate=locate)
                            tf.reset_default_graph()
                            nn.network()
                            nn.cost()
                            nn.trainer()
                            nn.initializer()     

                            with tf.Session() as session:
                                logdir = '/tmp/tensorflowlogs/X_sub_%s/11x11/pca_all_%s/rand_%s/cnn_%s/rnn_%s/locate_knn_%s/treat_%s/' % (subject_id,pca,rand_test,cnn,rnn,locate,lab_treat)
                                if tf.gfile.Exists(logdir):
                                    tf.gfile.DeleteRecursively(logdir)
                                tf.gfile.MakeDirs(logdir)
                                train_writer = tf.summary.FileWriter(logdir,session.graph)

                                session.run(nn.init_step)

                                for batch_num in range(0,batches):
                                    err_l_prev = 1000.
                                    err_l = 500.
                                    batch = batch_list[batch_num]
                                    print "Train batch ", batch_num, batch

                                    #pick a first batch of batch_size
                                    prepro = prepro_class.prepro(selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate,treat=treat,rnn=rnn,Wt=Wt)
                                    if subject_id is 'aud':
                                        prepro.aud_dataset()
                                    else:
                                        prepro.faces_dataset(subject_id)
                                    meas_img = prepro.meas_img_all
                                    qtrue = prepro.qtrue_all
                                    batch_size = prepro.total_batch_size
                                    step=0
                                    while step<per_batch:
                                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                        run_metadata = tf.RunMetadata()

                                        if rnn is False and cnn is not 'fft':
                                            train_summary, _ , guess,true,cost = session.run([nn.train_summary, nn.train_step, nn.qhat, nn.qtrain_unflat, nn.cost],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                        else:
                                            train_summary, _ , guess,true,cost = session.run([nn.train_summary, nn.train_step, nn.qhat_last, nn.qtrain_last, nn.cost],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})
                                        
                                        if step % print_step==0:
                                            print "Train Step: ", step, "Cost: ",cost

                                        if step % plot_step==0:

                                            if locate is True: locate=1
                                            if locate>0:
                                                pred_obs(guess, true, locate,name+str(step))
                                                

                                        writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'cost':cost})

                                        tstep=step+batch_num*per_batch
                                        train_writer.add_run_metadata(run_metadata, 'train_step%03d' % tstep)
                                        train_writer.add_summary(train_summary, tstep)

                                        if step % val_step==0 and step!=0:
                                            summary,guess,true,costv = session.run([nn.merged,nn.qhat,nn.qtrain_unflat, nn.cost], feed_dict={nn.qtrainPH: qtrue_val, nn.measPH: meas_img_val, nn.dropoutPH: dropout, nn.betaPH: beta})
                                            print "Val Step: ", step, "Cost: ",costv

                                            writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'cost':costv})
                                            train_writer.add_summary(summary, tstep)


                                        step+=1


                                #save_path = nn.saver.save(session, "./data/model.ckpt")
                                #print("Model saved in file: %s" % save_path)

                                #test batch
                                
                                if rnn is False and cnn is not 'fft':
                                    guess,true,costt = session.run([nn.qhat, nn.qtrain_unflat,nn.cost],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                else:
                                    guess,true,costt = session.run([nn.qhat_last, nn.qtrain_last,nn.cost],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                                print "Test Step: ", step, "Cost: ", costt
                                writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'cost':costt})

                                if locate is True: locate=1
                                #if locate>0:
                                    #pred_obs(guess, true, locate, name+str(-2))
                                    
                    csvfile.close()
