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
import time
###############################################################################

#meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True, subsample=10)


for train_id in [7,8]:
    for test_id in [7,8]:
        meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(train_id,locate=True)
        meas_dims, m, p, n_steps, total_batch_size_test = nn_prepro.faces_dataset(test_id,locate=True)
        a = np.arange(0,total_batch_size)
        a_test = np.arange(0,total_batch_size_test)

        test_size = int(.2*total_batch_size_test)
        val_size = int(.2*.8*total_batch_size)
        

        if train_id==test_id:
            test_size_train = test_size
            test = np.random.choice(a,test_size_train,replace=False)

            prob_select = np.ones(a.shape)/(total_batch_size-test_size_train)
            prob_select[test]=0.

            val = np.random.choice(a,val_size,replace=False,p=prob_select)
        else:
            test_size_train = 0
            test = np.random.choice(a_test,test_size,replace=False)
            val = np.random.choice(a,val_size,replace=False)

        
        print "Meas: ", m, " Out: ",p, " Steps: ",n_steps
        time.sleep(1)
        fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
        subsample = 1
        fname = './nn_real_locate_faces_cross_fix_tune_%s_%s.csv' % (train_id, test_id)
        with open(fname,'w') as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
            writer.writeheader()
            for cost in ['rmse']:
                for cost_step in ['last']:
                    for learning_rate in [0.005]:
                        for dropout in [1.0]:
                            for beta in [0.]:
                                for per_batch in [500]:
                                    for batch_size in [val_size]:
                                        print "Test size: ", test_size, " Val_size: ", val_size, " Batch size: ", batch_size, " Total size: ", total_batch_size
                                        for batches in [int(total_batch_size/batch_size)]:
                                            print "Batches: ", batches
                                            for k_conv in [3]:
                                                for n_conv1 in [2]:
                                                    for n_conv2 in [5]:
                                                        for n_layer in [1,2,3]:
                                                            n_chan_in=2
                                                            k_pool=1
                                                            n_out=p
                                                            n_in=meas_dims[0]*meas_dims[1]*2
                                                            n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2
                                                            for n_lstm in [10,25]:
                                                                val_step=int(per_batch/5)
                                                                if test_id==train_id:
                                                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                    #pick a test batch
                                                                    print "Test batch ",test
                                                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.faces_dataset(test_id,selection=val,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                    #pick a test batch
                                                                    print "Val batch ",val
                                                                    prob_select = np.ones(a.shape)/(total_batch_size-test_size_train-val_size)
                                                                    prob_select[test]=0.
                                                                    prob_select[val]=0.
                                                                    batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
                                                                    print "Train batch ", batch
                                                                    #pick a first batch of batch_size
                                                                    batch_num=0
                                                                    meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(train_id,selection=batch,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                else:
                                                                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                    #pick a test batch
                                                                    print "Test batch ",test
                                                                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.faces_dataset(train_id,selection=val,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                    #pick a test batch
                                                                    print "Val batch ",val
                                                                    prob_select = np.ones(a.shape)/(total_batch_size-test_size_train-val_size)
                                                                    prob_select[val]=0.
                                                                    batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
                                                                    print "Train batch ", batch
                                                                    #pick a first batch of batch_size
                                                                    batch_num=0
                                                                    meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(train_id,selection=batch,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)

                                                                cnn_rnn=tf_class.tf_meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cost_func=cost,cost_time=cost_step,beta=beta,cnn=True)
                                                                tf.reset_default_graph()
                                                                cnn_rnn.network()
                                                                with tf.Session() as session:

                                                                    session.run(cnn_rnn.init_step)
                                                                    step=0
                                                                    while step<per_batch and batch_num<batches:

                                                                        _ , guess,ce,acc,err,ce_l,acc_l,err_l  = session.run([cnn_rnn.train_step, cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                                             feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})

                                                                        if step % 10==0:
                                                                            print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                                                        writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})
                                                                        if step % val_step==0 and step!=0:
                                                                            guess,cev,accv,errv,ce_lv,acc_lv,err_lv = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                                                        feed_dict={cnn_rnn.qtruePH: qtrue_val, cnn_rnn.measPH: meas_img_val, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                                                            print "Val Step: ", step, "CE: ",cev, " Accuracy: ", accv, "RMSE: ", errv, "CE last: ",ce_lv, " Accuracy last: ", acc_lv, "RMSE last: ", err_lv

                                                                            writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'xentropy':cev,'rmse':errv,'accuracy':accv,'xentropy_last':ce_lv,'rmse_last':err_lv,'accuracy_last':acc_lv})
                                                                        step+=1
                                                                        if (step==(per_batch) and step!=0):#
                                                                            batch_num+=1

                                                                            #pick a nth batch of batch_size
                                                                            batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
                                                                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(train_id,selection=batch,pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)

                                                                            print "New batch ", batch_num, batch
                                                                            step=0

                                                                    #test batch
                                                                    guess,cet,acct,errt,ce_lt,acc_lt,err_lt = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                feed_dict={cnn_rnn.qtruePH: qtrue_test, cnn_rnn.measPH: meas_img_test, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                                                    print "Test Step: ", step, "CE: ",cet, " Accuracy: ", acct, "RMSE: ", errt, "CE last: ",ce_lt, " Accuracy last: ", acc_lt, "RMSE last: ", err_lt

                                                                    writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'xentropy':cet,'rmse':errt,'accuracy':acct,'xentropy_last':ce_lt,'rmse_last':err_lt,'accuracy_last':acc_lt})

        csvfile.close()
