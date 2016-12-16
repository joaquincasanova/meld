import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import tf_class
import csv

fieldnames=['fixed','noise','batches','learning rate','batch_size','per_batch','dropout','k_conv','n_conv1','n_conv2','n_layer','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

with open('./nn_last_errors_noise_free.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for fixed in [False]:
        for noise in [True]:
            for learning_rate in [0.005]:
                for batches in [10]:
                    for dropout in [.99]:
                        for per_batch in [500]:
                            for batch_size in [90]:
                                for k_conv in [3]:
                                    for n_conv1 in [2]:
                                        for n_conv2 in [7]:
                                            for n_layer in [2]:
                                                for n_steps in [100]:
                                                    print 'fixed',fixed,'noise',noise,'batches',batches,'learning rate',learning_rate,'batch_size',batch_size,'per_batch',per_batch,'dropout',dropout,'k_conv',k_conv,'n_conv1',n_conv1,'n_conv2',n_conv2,'n_layer',n_layer,'n_steps',n_steps
                                                    test_size=batch_size
                                                    if n_steps is None:
                                                        Ts = 1.0
                                                    else:
                                                        Ts = 1.0/n_steps#s, sample time
                                                    meas_dims=[11,11]
                                                    dipole_dims=[5,5,5]

                                                    n_chan_in=2
                                                    if fixed is True:
                                                        instance=dipole_class_xyz.dipole(Ts, batch_size, n_steps, n_chan_in, meas_dims,dipole_dims,orient=fixed,noise_flag=noise)
                                                    else:
                                                        instance=dipole_class_xyz.dipole(Ts, batch_size, n_steps, n_chan_in, meas_dims,dipole_dims,orient=fixed,noise_flag=noise)

                                                    instance.batch_sequence_gen()
                                                    #instance.fields_plot()
                                                    #instance.dipole_plot_scalar()
                                                    meas_img=instance.meas_img
                                                    qtrue=instance.qtrue
                                                    m=instance.m
                                                    p=instance.p
                                                    k_pool=1
                                                    n_out=p
                                                    n_in=meas_dims[0]*meas_dims[1]*2
                                                    n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2
                                                    n_lstm=n_steps
                                                    cnn_rnn=tf_class.tf_meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cost_func='rmse',cost_time='last')
                                                    tf.reset_default_graph()
                                                    cnn_rnn.network()

                                                    with tf.Session() as session:

                                                        session.run(cnn_rnn.init_step)

                                                        for step in range(0,per_batch*batches):
                                                            _ , guess,ce,acc,err,ce_l,acc_l,err_l  = session.run([cnn_rnn.train_step, cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                             feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout})

                                                            if step % 10==0:
                                                                print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                                                writer.writerow({'fixed':fixed,'noise':noise,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})
                                                            if step % per_batch ==0 and step!=0:#generate a new batch

                                                                if fixed is False:
                                                                    instance=dipole_class_xyz.dipole(Ts, batch_size, n_steps, n_chan_in, meas_dims,dipole_dims,orient=fixed,noise_flag=noise)

                                                                instance.batch_sequence_gen()
                                                                meas_img=instance.meas_img
                                                                qtrue=instance.qtrue

                                                        if fixed is False:
                                                            instance=dipole_class_xyz.dipole(Ts, batch_size, n_steps, n_chan_in, meas_dims,dipole_dims,orient=fixed,noise_flag=noise)

                                                        instance.batch_sequence_gen()
                                                        meas_img=instance.meas_img
                                                        qtrue=instance.qtrue
                                                        guess,ce,acc,err,ce_l,acc_l,err_l = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                        feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout})
                                                        print "Test Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                                        writer.writerow({'fixed':fixed,'noise':noise,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'train step':-1,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})
                                                        
                                                        
    csvfile.close()
