import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import meld_net
import csv
import nn_prepro
import time
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

###############################################################################

#meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True, subsample=10)

params_list = [[25,2,3,100,3,.2,.1,.1]]
params_list = [[3,3,7,100,3,.2,.1,.1]]
pca = True
rand_test = True
subsample = 1
locate = 1
for cnn in [True]:
    for rnn in [False]:
        for subject_id in ['aud']:
            print 'Subject: ',subject_id,' PCA: ',pca,' Random: ',rand_test, ' CNN: ',cnn, ' RNN: ',rnn, 'Locate: ',locate

            fieldnames=['batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','cost']
            fname = './data/subject_%s_pca_%s_rand_%s_cnn_%s_rnn_%s_locate_%s.csv' % (subject_id, pca, rand_test, cnn, rnn,locate)

            with open(fname,'a') as csvfile:
                writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()

                for [k_conv, n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:
                    if subject_id is 'aud':
                        meas_dims, m, p, n_steps, total_batch_size = nn_prepro.aud_dataset(justdims=True,cnn=cnn,locate=locate)
                    else:
                        meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(subject_id,cnn=cnn,justdims=True,locate=locate)

                    val_step=50
                    learning_rate = 0.005
                    dropout = 1.
                    beta = 0.
                    
                    if cnn is 'fft':
                        n_chan_in=1
                    else:
                        n_chan_in=2
                    
                    k_pool=1
                    n_out=p

                    test, val, batch_list, batches = nn_prepro.ttv(total_batch_size,test_frac,val_frac,batch_frac,rand_test=rand_test)

                    print "Meas: ", m, " Out: ",p, " Steps: ",n_steps

                    per_batch = int(5000/batches)

                    nn=meld_net.meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cnn=cnn,rnn=rnn,locate=locate)
                    tf.reset_default_graph()
                    nn.network()
                    nn.cost()
                    nn.trainer()
                    nn.initializer()     
                    if subject_id is 'aud':
                        meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.aud_dataset(selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    else:
                        meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(subject_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    #pick a test batch
                    print "Test batch ",test

                    if subject_id is 'aud':
                        meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.aud_dataset(selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    else:
                        meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.faces_dataset(subject_id,selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    #pick a val batch
                    print "Val batch ",val

                    with tf.Session() as session:
                        logdir = '/tmp/tensorflowlogs/sub_%s/11x11/pca_%s/rand_%s/cnn_%s/rnn_%s/locate_%s/' % (subject_id,pca,rand_test,cnn,rnn,locate)
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
                            if subject_id is 'aud':
                                meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.aud_dataset(selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                            else:
                                meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(subject_id,selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)

                            step=0
                            while step<per_batch:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()

                                train_summary, _ , guess,cost = session.run([nn.train_summary, nn.train_step, nn.qhat, nn.cost],feed_dict={nn.qtrainPH: qtrue, nn.measPH: meas_img, nn.dropoutPH: dropout, nn.betaPH: beta})


                                if step % 10==0:
                                    print "Train Step: ", step, "Cost: ",cost

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


                        save_path = nn.saver.save(session, "./data/model.ckpt")
                        print("Model saved in file: %s" % save_path)

                        #test batch
                        guess,guess_last,true,true_last,costt = session.run([nn.qhat,nn.qhat_last, nn.qtrain_unflat,nn.qtrain_last,nn.cost],feed_dict={nn.qtrainPH: qtrue_test, nn.measPH: meas_img_test, nn.dropoutPH: dropout, nn.betaPH: beta})
                        print "Test Step: ", step, "Cost: ", costt
                        writer.writerow({'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'cost':costt})

                        if locate is True: locate=1
                        if locate>0:
                            if rnn is False:                     
                                for l in range(0,locate):
                                    z = np.squeeze(guess[:,2+l*3])
                                    y = np.squeeze(guess[:,1+l*3])
                                    x = np.squeeze(guess[:,0+l*3])

                                    zt = np.squeeze(true[:,2+l*3])
                                    yt = np.squeeze(true[:,1+l*3])
                                    xt = np.squeeze(true[:,0+l*3])

                                    fig = plt.figure()
                                    ax = fig.gca(projection='3d')
                                    ax.plot(x, y, z, 'ob')
                                    ax.plot(xt, yt, zt, 'or')
                                    plt.title('Number '+str(l))
                                    plt.show()

                                    plt.subplot(3, 1, 1)
                                    plt.plot(xt, x, 'o')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('X of '+str(l))

                                    plt.subplot(3, 1, 2)
                                    plt.plot(yt,y,'o')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('Y of '+str(l))

                                    plt.subplot(3, 1, 3)
                                    plt.plot(zt,z,'o')
                                    plt.xlabel('True (mm)')
                                    plt.ylabel('Predicted (mm)')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('Z of '+str(l))
                                    plt.show()                            
                            else:                     
                                for l in range(0,locate):
                                    z = np.squeeze(guess_last[:,2+l*3])
                                    y = np.squeeze(guess_last[:,1+l*3])
                                    x = np.squeeze(guess_last[:,0+l*3])

                                    zt = np.squeeze(true_last[:,2+l*3])
                                    yt = np.squeeze(true_last[:,1+l*3])
                                    xt = np.squeeze(true_last[:,0+l*3])

                                    fig = plt.figure()
                                    ax = fig.gca(projection='3d')
                                    ax.plot(x, y, z, 'ob')
                                    ax.plot(xt, yt, zt, 'or')
                                    plt.title('Number '+str(l))
                                    plt.show()

                                    plt.subplot(3, 1, 1)
                                    plt.plot(xt, x, 'o')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('X of '+str(l))

                                    plt.subplot(3, 1, 2)
                                    plt.plot(yt,y,'o')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('Y of '+str(l))

                                    plt.subplot(3, 1, 3)
                                    plt.plot(zt,z,'o')
                                    plt.xlabel('True (mm)')
                                    plt.ylabel('Predicted (mm)')
                                    plt.xlim(-100,100)
                                    plt.ylim(-100,100)
                                    plt.title('Z of '+str(l))
                                    plt.show()

            csvfile.close()
