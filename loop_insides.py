val_step=100
                                                                meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(subject_id,selection=range(0,test_size),pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                #pick a test batch
                                                                print "Test batch"
                                                                print 0, test_size
                                                                #pick a val batch
                                                                meas_img_val, qtrue_val, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(subject_id,selection=range((test_size),(test_size+val_size)),pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                print "Val batch"
                                                                print test_size, test_size+val_size
                                                                #pick a first batch of batch_size
                                                                batch_num=0
                                                                choose = np.random.choice(total_batch_size-test_size-val_size,batch_size,replace=False)
                                                                meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(subject_id,selection=(test_size+val_size+choose),pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                batch_num = 0
                                                                print "New batch", batch_num
                                                                print choose
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
                                                                            print "New batch", batch_num

                                                                            choose = np.random.choice(total_batch_size-test_size-val_size,batch_size,replace=False)
                                                                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(subject_id,selection=(test_size+val_size+choose),pca=True,subsample=subsample,justdims=False,cnn=True,locate=True)
                                                                            print choose

                                                                            step=0

                                                                    #test batch
                                                                    guess,cet,acct,errt,ce_lt,acc_lt,err_lt = session.run([cnn_rnn.qhat, cnn_rnn.cross, cnn_rnn.accuracy,cnn_rnn.rmse, cnn_rnn.cross_last, cnn_rnn.accuracy_last,cnn_rnn.rmse_last],
                                                                                feed_dict={cnn_rnn.qtruePH: qtrue_test, cnn_rnn.measPH: meas_img_test, cnn_rnn.dropoutPH: dropout, cnn_rnn.betaPH: beta})
                                                                    print "Test Step: ", step, "CE: ",cet, " Accuracy: ", acct, "RMSE: ", errt, "CE last: ",ce_lt, " Accuracy last: ", acc_lt, "RMSE last: ", err_lt

                                                                    writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'xentropy':cet,'rmse':errt,'accuracy':acct,'xentropy_last':ce_lt,'rmse_last':err_lt,'accuracy_last':acc_lt})
