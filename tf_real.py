import numpy as np
from numpy import matlib
import sphere
import dipole_class_xyz
import tensorflow as tf
import csv
import meas_class
import nn_prepro
import time

# Create model
def conv2d(img, w, b, name):
    return tf.nn.relu(tf.nn.bias_add\
                      (tf.nn.conv2d(img, w,\
                                    strides=[1, 1, 1, 1],\
                                    padding='VALID'),b),name=name)

def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding='VALID')

def network(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=None,n_lstm=None,n_layer=None,cost_func='cross',cost_time='all', beta=0.0, cnn=True):

    learning_rate=learning_rate
    print "learning rate: ", learning_rate
    meas_dims=meas_dims
    print "meas_dims: ", meas_dims
    k_conv=k_conv
    print "kconv: ", k_conv
    k_pool=k_pool
    print "k_pool: ", k_pool
    n_chan_in=n_chan_in
    print "n_chan_in: ", n_chan_in
    n_conv1=n_conv1
    print "n_conv1: ", n_conv1
    n_conv2=n_conv2
    print "n_conv2: ", n_conv2
    n_out=n_out
    print "n_out: ", n_out 
    if n_steps is 1:
        n_steps=None
    else:
        n_steps=n_steps
    print "n_steps: ", n_steps
    n_lstm=n_lstm
    print "n_lstm: ", n_lstm
    n_layer=n_layer
    print "n_layer: ", n_layer
    cost_func=cost_func
    print "cost_func: ", cost_func
    cost_time=cost_time
    print "cost_time: ", cost_time
    beta=beta
    print "beta: ", beta
    cnn=cnn
    print cnn
    if cnn is True:
        n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2
    else:
        n_dense=meas_dims
    print "n_dense: ", n_dense

    dropoutPH = tf.placeholder(tf.float32, name="dropout")
    betaPH =  tf.placeholder(tf.float32, name="beta")
    if n_out==3:
        std=10.
    else:
        std=0.1

    if cnn is True:
        if  n_steps is None:
            measPH=tf.placeholder(tf.float32,shape=(None,meas_dims[0],meas_dims[1],n_chan_in), name="meas")
            qtruePH=tf.placeholder(tf.float32,shape=(None, n_out), name="qtrue")
        else:
            measPH=tf.placeholder(tf.float32,shape=(None, n_steps, meas_dims[0], meas_dims[1],n_chan_in), name="meas")
            qtruePH=tf.placeholder(tf.float32,shape=(None, n_steps, n_out), name="qtrue")
        # Store layers weight & bias

        with tf.name_scope('convolutional_layer_1'):
            wc1 = tf.Variable(tf.random_normal([k_conv, k_conv, n_chan_in, n_conv1])) # kxk conv, 2 input, n_conv outputs
            bc1 = tf.Variable(tf.random_normal([n_conv1]))

            if n_steps is None:
                conv1 = conv2d(measPH,wc1,bc1,name="conv1")
            else:
                #reshape to fold batch and timestep indices, needed for compatibility with conv2d
                measfold = tf.reshape(measPH,[-1,meas_dims[0], meas_dims[1], n_chan_in])
                conv1 = conv2d(measfold,wc1,bc1,name="conv1")

            conv1 = max_pool(conv1,k=k_pool)

            # Apply Dropout
            conv1 = tf.nn.dropout(conv1,dropoutPH)

        with tf.name_scope('convolutional_layer_2'):
            wc2 = tf.Variable(tf.random_normal([k_conv, k_conv, n_conv1, n_conv2])) # kxk conv, 2 input, n_conv outputs
            bc2 = tf.Variable(tf.random_normal([n_conv2]))
            conv2 = conv2d(conv1,wc2,bc2,name="conv2" )
            # Apply Dropout
            conv2 = tf.nn.dropout(conv2, dropoutPH)

        with tf.name_scope('dense_layer'):
            if n_steps is None:
                wd = tf.Variable(tf.random_normal([n_dense, n_out]),stddev=std) # fully connected, image inputs, n_out outputs
                bd = tf.Variable(tf.random_normal([n_out]),stddev=std)
                dense = tf.reshape(conv2, [-1, n_dense]) # Reshape conv2 output to fit dense layer input
                logits = tf.add(tf.matmul(dense,wd),bd)#logits

            else:
                dense = tf.reshape(conv2, [-1, n_dense]) # Reshape conv1 output to fit dense layer input
                wd = tf.Variable(tf.truncated_normal([n_dense, n_lstm], stddev=0.1))
                bd = tf.Variable(tf.constant(0.1, shape=[n_lstm]))
                dense_out = tf.nn.softmax(tf.matmul(dense, wd) + bd,name="dense_out")
                dense_out = tf.nn.dropout(dense_out, dropoutPH)
    else:#NO CNN
        if  n_steps is None:
            measPH=tf.placeholder(tf.float32,shape=(None,meas_dims), name="meas")
            qtruePH=tf.placeholder(tf.float32,shape=(None, n_out), name="qtrue")
        else:
            measPH=tf.placeholder(tf.float32,shape=(None, n_steps, meas_dims), name="meas")
            qtruePH=tf.placeholder(tf.float32,shape=(None, n_steps, n_out), name="qtrue")

        with tf.name_scope('dense_layer'):
            if n_steps is None:
                wd = tf.Variable(tf.random_normal([meas_dims, n_out]),stddev=std) # fully connected, image inputs, n_out outputs
                bd = tf.Variable(tf.random_normal([n_out]),stddev=std)
                logits = tf.add(tf.matmul(measPH,wd),bd)#logits
            else:
                dense = tf.reshape(measPH, [-1, meas_dims]) # Reshape input to fit dense layer input
                wd = tf.Variable(tf.truncated_normal([n_dense, n_lstm], stddev=0.1))
                bd = tf.Variable(tf.constant(0.1, shape=[n_lstm]))
                dense_out = tf.nn.relu(tf.matmul(dense, wd) + bd,name="dense_out")#try relu
                dense_out = tf.nn.dropout(dense_out, dropoutPH)

    with tf.name_scope('rnn_layer'):
        if n_steps is not None:
            #now predict sequence of firing
            cell = tf.nn.rnn_cell.BasicLSTMCell(n_lstm,state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropoutPH)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layer,state_is_tuple=True)  

            data = tf.reshape(dense_out, [-1, n_steps, n_lstm])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

            val = tf.transpose(output,[1,0,2])#nxbxn_lstm
            last = tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm

            outs = tf.reshape(output, [-1, n_lstm])#n*bxn_lstm

            wrnn = tf.Variable(tf.random_normal([n_lstm,n_out], stddev=std))
            brnn = tf.Variable(tf.random_normal([n_out],stddev=std))

            logits = tf.add(tf.matmul(outs,wrnn),brnn)#logits - b*nxp
            #logits = tf.reshape(logits,[-1,n_out])#b*nxp

            logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

    with tf.name_scope('cost'):
        if n_steps is not None:
            if n_out==3:
                qhat = logits
                qhat_last = logits_last                
            else:
                qhat = tf.nn.softmax(logits,name="qhat")
                qhat_last = tf.nn.softmax(logits_last,name="qhat_last")
        else:
                if n_out==3:
                    qhat = logits
                else:
                    qhat = tf.nn.softmax(logits,name="qhat")

        if cnn is True:
            reg=tf.multiply(betaPH,
                            tf.add(tf.add(tf.nn.l2_loss(wd),tf.nn.l2_loss(wc1)),
                                   tf.nn.l2_loss(wc2)))
        else:
            reg=tf.multiply(betaPH,tf.nn.l2_loss(wd))

        A=tf.argmax(logits,1)
        if n_steps is None:

            B=tf.argmax(qtruePH,1)
            qtrue_OH = tf.one_hot(B,n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
            cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, qtruePH_OH),name="cross"),reg)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
            rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qhat,qtruePH))),name="rmse"),reg)

            cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, qtruePH_OH),name="cross_last"),reg)        
            accuracy_last = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy_last")
            rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qhat,qtruePH))),name="rmse_last"),reg)
        else:
            qtrue_unflat = tf.reshape(qtruePH,[-1,n_out])#b*nxp
            AA=tf.argmax(logits_last,1)
            B=tf.argmax(qtrue_unflat,1)#b*nx1
            qtrue_OH = tf.one_hot(B,n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation

            qtrue_tran = tf.transpose(qtruePH,[1,0,2])#nxbxp
            qtrue_last = tf.gather(qtrue_tran, int(qtrue_tran.get_shape()[0])-1)#bxp
            BB=tf.argmax(qtrue_last,1)#bx1
            qtrue_last_OH = tf.one_hot(BB,n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation

            cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, qtrue_OH),name="cross"),reg)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
            rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qhat,qtrue_unflat))),name="rmse"),reg)

            cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_last, qtrue_last_OH),name="cross_last"),reg)
            accuracy_last = tf.reduce_mean(tf.cast(tf.equal(AA,BB),tf.float32),name="accuracy_last")
            rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qtrue_last,qhat_last))),name="rmse_last"),reg)

    with tf.name_scope('train_step'):
        #use rmse as cost function if you are trying to fit current density OR location.
        if n_steps is None:
            if cost_func=='cross':
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
            elif cost_func=='rmse':
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse)
            else:
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
        else:
            if cost_time=='all':
                if cost_func=='cross':
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross)
                elif cost_func=='rmse':
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse)
                else:
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse)
            elif cost_time=='last':
                if cost_func=='cross':
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_last)
                elif cost_func=='rmse':
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse_last)
                else:
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse_last)
            else:
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse_last)
        saver = tf.train.Saver()

    init_step = tf.global_variables_initializer()

    return init_step, train_step, saver, qhat, cross, accuracy, rmse, cross_last, accuracy_last,rmse_last,qtruePH, measPH, dropoutPH, betaPH
###############################################################################

#meas_img_all, qtrue_all, meas_dims, m, p, n_steps, total_batch_size=nn_prepro.aud_dataset(pca=True, subsample=10)

params_list = [[2,5,10,1,.2,.1,.1],[3,7,15,3,.2,.1,.1]]

for [pca, rand_test] in [[True, True]]:
    cnn=False
    locate=False
    subsample = 20

    cost  = 'cros's
    cost_step = 'last'            
    for train_id in [7]:
        for test_id in [7]:
            print 'Train on: ',train_id,' Test on: ',test_id,' PCA: ',pca,' Random: ',rand_test
            fieldnames=['cost','cost_step','batches','learning rate','batch_size','per_batch','dropout','beta','k_conv','n_conv1','n_conv2','n_layer','n_lstm','n_steps','train step','xentropy','rmse','accuracy','xentropy_last','rmse_last','accuracy_last']
            
            fname = './data/nn_real_nocnn_%s_%s_pca_%s_rand_%s.csv' % (train_id, test_id, pca, rand_test)
            with open(fname,'a') as csvfile:
                writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()

                for [n_conv1, n_conv2, n_lstm, n_layer, test_frac, val_frac, batch_frac] in params_list:
                    meas_dims, m, p, n_steps, total_batch_size = nn_prepro.faces_dataset(train_id,subsample=subsample,locate=locate)
                    meas_dims, m, p, n_steps, total_batch_size_test = nn_prepro.faces_dataset(test_id,subsample=subsample,locate=locate)

                    a = np.arange(0,total_batch_size)
                    a_test = np.arange(0,total_batch_size_test)

                    #halt criteria
                    delta_v_err_halt = 1.
                    delta_err_halt = 2e-5

                    val_step=50
                    learning_rate = 0.001
                    dropout = 1.0
                    beta = 0.
                    k_conv = 3
                    n_chan_in=2
                    k_pool=1

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

                    
                    assert np.intersect1d(test,val).size is 0

                    print "Test size: ", test_size, " Val_size: ", val_size, " Batch size: ", batch_size, " Total size: ", total_batch_size
                    print "Meas: ", m, " Out: ",p, " Steps: ",n_steps

                    batches = int((total_batch_size-val_size-test_size_train)/batch_size)
                    
                    per_batch = int(5000/batches)
                    print "Batches: ", batches, " Batches*batch_size: ", batches*batch_size, " Train set size: ",(total_batch_size-val_size-test_size_train), " Per batch: ", per_batch

                    n_in=meas_dims[0]*meas_dims[1]*2
                    n_dense=int((meas_dims[0]-k_conv+1)/k_pool-k_conv+1)*int((meas_dims[1]-k_conv+1)/k_pool-k_conv+1)*n_conv2

                    time.sleep(1)
                    batch_num=0

                    meas_img_test, qtrue_test, meas_dims, m, p, n_steps, test_size = nn_prepro.faces_dataset(test_id,selection=test,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    #pick a test batch
                    print "Test batch ",test

                    meas_img_val, qtrue_val, meas_dims, m, p, n_steps, val_size = nn_prepro.faces_dataset(train_id,selection=val,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)
                    #pick a val batch
                    print "Val batch ",val

                    err_lv_prev = 1000.
                    err_lv = 500.

                    n_out=p

                    tf.reset_default_graph()
                    
                    init_step, train_step, saver, qhat, cross, accuracy, rmse, cross_last, accuracy_last,rmse_last,qtruePH, measPH, dropoutPH, betaPH = network(learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=n_steps,n_lstm=n_lstm,n_layer=n_layer,cost_func=cost,cost_time=cost_step, beta=beta, cnn=cnn)

                    with tf.Session() as session:

                        session.run(init_step)

                        for batch_num in range(0,batches):
                            err_l_prev = 1000.
                            err_l = 500.
                            if rand_test is True:
                                batch=np.random.choice(a,batch_size,replace=False,p=prob_select)
                            else:
                                choose = np.arange(batch_size)
                                batch=test_size_train+val_size+batch_num*batch_size+choose
                            
                            assert np.intersect1d(batch,test).size is 0
                            assert np.intersect1d(batch,val).size is 0

                            print "Train batch ", batch_num, batch
                            #pick a first batch of batch_size
                            meas_img, qtrue, meas_dims, m, p, n_steps, batch_size = nn_prepro.faces_dataset(train_id,selection=batch,pca=pca,subsample=subsample,justdims=False,cnn=cnn,locate=locate)

                            prob_select*=float(total_batch_size-test_size_train-val_size-batch_size*batch_num)/float(total_batch_size-test_size_train-val_size-batch_size*(batch_num+1))
                            prob_select[batch]=0.

                            step=0
                            while step<per_batch:# and abs(err_l-err_l_prev)/err_l_prev>delta_err_halt:
                                err_l_prev=err_l

                                _ , guess,ce,acc,err,ce_l,acc_l,err_l = session.run([train_step, qhat, cross, accuracy,rmse, cross_last, accuracy_last,rmse_last],
                                                                                     feed_dict={qtruePH: qtrue, measPH: meas_img, dropoutPH: dropout, betaPH: beta})
                                

                                if step % 10==0:
                                    print "Train Step: ", step, "CE: ",ce, " Accuracy: ", acc, "RMSE: ", err, "CE last: ",ce_l, " Accuracy last: ", acc_l, "RMSE last: ", err_l
                                    
                                writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_steps':n_steps,'n_lstm':n_lstm,'train step':step,'xentropy':ce,'rmse':err,'accuracy':acc,'xentropy_last':ce_l,'rmse_last':err_l,'accuracy_last':acc_l})



                                if step % val_step==0 and step!=0:

                                    err_lv_prev=err_lv

                                    guess,cev,accv,errv,ce_lv,acc_lv,err_lv = session.run([qhat, cross, accuracy,rmse, cross_last, accuracy_last,rmse_last],
                                                                                feed_dict={qtruePH: qtrue_val, measPH: meas_img_val, dropoutPH: dropout, betaPH: beta})
                                    print "Val Step: ", step, "CE: ",cev, " Accuracy: ", accv, "RMSE: ", errv, "CE last: ",ce_lv, " Accuracy last: ", acc_lv, "RMSE last: ", err_lv
                                    #print "Val Error change: ",err_lv,(err_lv-err_lv_prev)/err_lv_prev

                                    #if (err_lv-err_lv_prev)/err_lv_prev>delta_v_err_halt:
                                    #    break
                                    
                                    writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-1,'xentropy':cev,'rmse':errv,'accuracy':accv,'xentropy_last':ce_lv,'rmse_last':err_lv,'accuracy_last':acc_lv})

                                step+=1
                                    
                        save_path = saver.save(session, "./data/model.ckpt")
                        print("Model saved in file: %s" % save_path)

                        #test batch
                        guess,cet,acct,errt,ce_lt,acc_lt,err_lt = session.run([qhat, cross, accuracy,rmse, cross_last, accuracy_last,rmse_last],
                                    feed_dict={qtruePH: qtrue_test, measPH: meas_img_test, dropoutPH: dropout, betaPH: beta})
                        print "Test Step: ", step, "CE: ",cet, " Accuracy: ", acct, "RMSE: ", errt, "CE last: ",ce_lt, " Accuracy last: ", acc_lt, "RMSE last: ", err_lt

                        writer.writerow({'cost':cost,'cost_step':cost_step,'batches':batches,'learning rate':learning_rate,'batch_size':batch_size,'per_batch':per_batch,'dropout':dropout,'beta':beta,'k_conv':k_conv,'n_conv1':n_conv1,'n_conv2':n_conv2,'n_layer':n_layer,'n_lstm':n_lstm,'n_steps':n_steps,'train step':-2,'xentropy':cet,'rmse':errt,'accuracy':acct,'xentropy_last':ce_lt,'rmse_last':err_lt,'accuracy_last':acc_lt})

            csvfile.close()
