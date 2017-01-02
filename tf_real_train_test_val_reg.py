import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import tf_class
import csv
import meas_class
#import mne
#from mne.datasets import sample
#from mne.minimum_norm import (make_inverse_operator, apply_inverse,
#                              write_inverse_operator, apply_inverse_epochs)
import nn_prepro
###############################################################################

meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True)

fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']

with open('./nn_real_pca_lstm.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for cost in ['cross']:
        for cost_step in ['last']:
            for learning_rate in [0.001]:
                for batches in [10]:
                    for dropout in [1.0]:
                        for beta in [0.]:
                            for per_batch in [500]:
                                for batch_size in [100]:
                                    total_batch_size-batches*batch_size
                                    test_size = 24
                                    val_size = 25
                                    print test_size, val_size, batch_size, total_batch_size
                                    for k_conv in [3]:
                                        for n_conv1 in [3]:
                                            for n_conv2 in [5]:
                                                for n_layer in [1,2]:
                                                    n_chan_in=2
                                                    k_pool=1
                                                    n_out=p
                                                    n_in=meas_dims[0]*meas_dims[1]*2
                                                    n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2
                                                    for n_lstm in [10,100,300]:
                                                        val_step=100

                                                        #pick a test batch
                                                        meas_img_test = meas_img_all[0:test_size,:,:,:,:]
                                                        qtrue_test = qtrue_all[0:test_size,:,:]
                                                        print "Test batch"
                                                        print 0, test_size
                                                        #pick a val batch
                                                        meas_img_val = meas_img_all[(test_size):(test_size+val_size),:,:,:,:]
                                                        qtrue_val = qtrue_all[(test_size):(test_size+val_size),:,:]
                                                        print "Val batch"
                                                        print test_size, test_size+val_size
                                                        #pick a first batch of batch_size
                                                        batch_num=0
                                                        choose = np.random.choice(total_batch_size-test_size-val_size,batch_size,replace=False)
                                                        meas_img = meas_img_all[(test_size+val_size+choose),:,:,:,:]
                                                        qtrue = qtrue_all[(test_size+val_size+choose),:,:]
                                                        batch_num = 0
                                                        print "New batch", batch_num
                                                        print choose
                                                        cnn_rnn=tf_class.tf_meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cost_func=cost,cost_time=cost_step,beta=beta)
                                                        tf.reset_default_graph()
                                                        cnn_rnn.network()
                                                        with tf.Session() as session:

                                                            session.run(cnn_rnn.init_step)
                                                            acc_lv=0
                                                            acc_lv_prev=0
                                                            step=0
                                                            while step<per_batch and batch_num<batches:

                                                                _ , guess,ce,acc,err,ce_l,acc_l,err_l  = session.run([cnn_rnn.train_step, cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                                     feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})

                                                                if step % 10==0:
                                                                    print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                                                writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})
                                                                if step % val_step==0 and step!=0:
                                                                    acc_lv_prev=acc_lv
                                                                    guess,cev,accv,errv,ce_lv,acc_lv,err_lv = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                                feed_dict={cnn_rnn.qtruePH: qtrue_val, cnn_rnn.measPH: meas_img_val, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                                                    print "Val Step: ", step, "CE: ",cev, " Accuracy: ", accv, "RMSE: ", errv, "CE last: ",ce_lv, " Accuracy last: ", acc_lv, "RMSE last: ", err_lv

                                                                    writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'xentropy':cev,'rmse':errv,'accuracy':accv,'xentropy_last':ce_lv,'rmse_last':err_lv,'accuracy_last':acc_lv})
                                                                step+=1
                                                                if (step==per_batch and step!=0) or (acc_lv-acc_lv_prev)<-.1 or acc_l>.95:#
                                                                    batch_num+=1
                                                                    print (acc_lv-acc_lv_prev)
                                                                    #pick a nth batch of batch_size
                                                                    print "New batch", batch_num

                                                                    choose = np.random.choice(total_batch_size-test_size-val_size,batch_size,replace=False)
                                                                    meas_img = meas_img_all[(test_size+val_size+choose),:,:,:,:]
                                                                    qtrue = qtrue_all[(test_size+val_size+choose),:,:]
                                                                    print choose
                                                                    
                                                                    step=0
                                                                    acc_lv=0
                                                                    acc_lv_prev=0

                                                            #test batch
                                                            guess,cet,acct,errt,ce_lt,acc_lt,err_lt = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                        feed_dict={cnn_rnn.qtruePH: qtrue_test, cnn_rnn.measPH: meas_img_test, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                                            print "Test Step: ", step, "CE: ",cet, " Accuracy: ", acct, "RMSE: ", errt, "CE last: ",ce_lt, " Accuracy last: ", acc_lt, "RMSE last: ", err_lt

                                                            writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})

    csvfile.close()
