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

params_list = [[2,5,10,1,.2,.1,.9]]#,[2,5,10,1,.2,.1,.1]]#,[3,7,15,3,.2,.1,.9]

for [pca, rand_test] in [['none', False],[True, False],[False, False]]:
    for train_id in [7]:
        for test_id in [7]:
            print 'Train on: ',train_id,' Test on: ',test_id,' PCA: ',pca,' Random: ',rand_test
            fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
            subsample = 1
            fname = './data/nn_real_relu_%s_%s_pca_%s_rand_%s.csv' % (train_id, test_id, pca, rand_test)
            with open(fname,'a') as csvfile:
                writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()

                for [n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:
                    meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(train_id,locate=True)
                    meas_dims, m, p, n_steps, total_batch_size_test = nn_prepro.faces_dataset(test_id,locate=True)

                    a = np.arange(0,total_batch_size)
                    a_test = np.arange(0,total_batch_size_test)

                    #halt criteria
                    delta_v_err_halt = 1.
                    delta_err_halt = 2e-5

                    val_step=50
                    cost  = 'rmse'
                    cost_step = 'last'
                    learning_rate = 0.001
                    dropout = 1.0
                    beta = 0.
                    k_conv = 3
                    n_chan_in=2
                    k_pool=1
                    n_out=p

                    test_size = int(test_frac*total_batch_size_test)

                    #pick test/val sets
                    if train_id==test_id:
                        test_size_train = test_size             
                        val_size = int(val_frac*(total_batch_size-test_size_train))             
                        batch_size = int(batch_frac*(total_batch_size-test_size_train))

                        if rand_test is True:
                            test = np.random.choice(a,test_size_train,replace=False)
                        else:
                            test = np.arange(0,test_size)

                        prob_select = np.ones(a.shape)/float(total_batch_size-test_size_train)
                        prob_select[test]=0.

                        #print "Prob select: ", prob_select

                        if rand_test is True:                     
                            val = np.random.choice(a,val_size,replace=False,p=prob_select)
                        else:
                            val = np.arange(test_size_train,test_size_train+val_size)

                        prob_select[val]=0.
                        prob_select*=float(total_batch_size-test_size_train)/float(total_batch_size-test_size_train-val_size)

                        #print "Prob select: ", prob_select

                    else:
                        test_size_train = 0                
                        val_size = int(val_frac*(total_batch_size-test_size_train))
                        batch_size = int(batch_frac*(total_batch_size-test_size_train))

                        prob_select = np.ones(a.shape)/float(total_batch_size-test_size_train)

                        #print "Prob select: ", prob_select

                        if rand_test is True:
                            test = np.random.choice(a_test,test_size,replace=False)
                            val = np.random.choice(a,val_size,replace=False)
                        else:
                            test = range(0,test_size)
                            val = range(0,val_size)

                        prob_select[val]=0.
                        prob_select*=float(total_batch_size-test_size_train)/float(total_batch_size-test_size_train-val_size)

                        #print "Prob select: ", prob_select


                    print "Test size: ", test_size, " Val_size: ", val_size, " Batch size: ", batch_size, " Total size: ", total_batch_size
                    print "Meas: ", m, " Out: ",p, " Steps: ",n_steps

                    batches = int((total_batch_size-val_size-test_size_train)/batch_size)
                    
                    per_batch = int(5000/batches)
                    print "Batches: ", batches, " Batches*batch_size: ", batches*batch_size, " Train set size: ",(total_batch_size-val_size-test_size_train), " Per batch: ", per_batch

                    n_in=meas_dims[0]*meas_dims[1]*2
                    n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2

                    time.sleep(1)
                    batch_num=0

                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=True,locate=True)
                    #pick a test batch
                    print "Test batch ",test

                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.faces_dataset(train_id,selection=val,pca=pca,subsample=subsample,justdims=False,cnn=True,locate=True)
                    #pick a val batch
                    print "Val batch ",val

                    cnn_rnn=tf_class.tf_meld(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps,n_lstm,n_layer,cost_func=cost,cost_time=cost_step,beta=beta,cnn=True)
                    tf.reset_default_graph()
                    cnn_rnn.network()
                    err_lv_prev = 1000.
                    err_lv = 500.
                    with tf.Session() as session:

                        session.run(cnn_rnn.init_step)

                        for batch_num in range(0,batches):
                            err_l_prev = 1000.
                            err_l = 500.
                            if rand_test is True:
                                batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
                            else:
                                choose = np.arange(batch_size)
                                batch=test_size_train+val_size+batch_num*batch_size+choose
                                
                            print "Train batch ", batch_num, batch
                            #pick a first batch of batch_size
                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(train_id,selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=True,locate=True)

                            prob_select*=float(total_batch_size-test_size_train-val_size-batch_size*batch_num)/float(total_batch_size-test_size_train-val_size-batch_size*(batch_num+1))
                            prob_select[batch]=0.

                            step=0
                            while step<per_batch:# and abs(err_l-err_l_prev)/err_l_prev>delta_err_halt:
                                err_l_prev=err_l

                                _ , guess,ce,acc,err,ce_l,acc_l,err_l  = session.run([cnn_rnn.train_step, cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                     feed_dict={cnn_rnn.qtruePH: qtrue, cnn_rnn.measPH: meas_img, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                

                                if step % 10==0:
                                    print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                    #print "Error change: ",err_l,abs(err_l-err_l_prev)/err_l_prev
                                writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})


                                if step % val_step==0 and step!=0:

                                    err_lv_prev=err_lv

                                    guess,cev,accv,errv,ce_lv,acc_lv,err_lv = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                feed_dict={cnn_rnn.qtruePH: qtrue_val, cnn_rnn.measPH: meas_img_val, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                    print "Val Step: ", step, "CE: ",cev, " Accuracy: ", accv, "RMSE: ", errv, "CE last: ",ce_lv, " Accuracy last: ", acc_lv, "RMSE last: ", err_lv
                                    #print "Val Error change: ",err_lv,(err_lv-err_lv_prev)/err_lv_prev

                                    #if (err_lv-err_lv_prev)/err_lv_prev>delta_v_err_halt:
                                    #    break
                                    
                                    writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'xentropy':cev,'rmse':errv,'accuracy':accv,'xentropy_last':ce_lv,'rmse_last':err_lv,'accuracy_last':acc_lv})

                                step+=1

                        #test batch
                        guess,cet,acct,errt,ce_lt,acc_lt,err_lt = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                    feed_dict={cnn_rnn.qtruePH: qtrue_test, cnn_rnn.measPH: meas_img_test, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                        print "Test Step: ", step, "CE: ",cet, " Accuracy: ", acct, "RMSE: ", errt, "CE last: ",ce_lt, " Accuracy last: ", acc_lt, "RMSE last: ", err_lt

                        writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'xentropy':cet,'rmse':errt,'accuracy':acct,'xentropy_last':ce_lt,'rmse_last':err_lt,'accuracy_last':acc_lt})

            csvfile.close()
