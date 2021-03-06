
import tensorflow as tf

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

class tf_meld:
    def __init__(self,learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=None,n_lstm=None,n_layer=None,cost_func='cross',cost_time='all', beta=0.0, cnn=True):
        self.learning_rate=learning_rate
        print "learning rate: ", self.learning_rate
        self.meas_dims=meas_dims
        print "meas_dims: ", self.meas_dims
        self.k_conv=k_conv
        print "kconv: ", self.k_conv
        self.k_pool=k_pool
        print "k_pool: ", self.k_pool
        self.n_chan_in=n_chan_in
        print "n_chan_in: ", self.n_chan_in
        self.n_conv1=n_conv1
        print "n_conv1: ", self.n_conv1
        self.n_conv2=n_conv2
        print "n_conv2: ", self.n_conv2
        self.n_out=n_out
        print "n_out: ", self.n_out 
        if n_steps is 1:
            self.n_steps=None
        else:
            self.n_steps=n_steps
        print "n_steps: ", self.n_steps
        self.n_lstm=n_lstm
        print "n_lstm: ", self.n_lstm
        self.n_layer=n_layer
        print "n_layer: ", self.n_layer
        self.cost_func=cost_func
        print "cost_func: ", self.cost_func
        self.cost_time=cost_time
        print "cost_time: ", self.cost_time
        self.beta=beta
        print "beta: ", self.beta
        self.cnn=cnn
        print self.cnn
        if self.cnn is True:
            self.n_dense=int((self.meas_dims[0]-self.k_conv+1)/self.k_pool-self.k_conv+1)*int((self.meas_dims[1]-self.k_conv+1)/self.k_pool-self.k_conv+1)*self.n_conv2
        else:
            self.n_dense=self.meas_dims
        print "n_dense: ", self.n_dense
    def network(self):
        
        self.dropoutPH = tf.placeholder(tf.float32, name="dropout")
        self.betaPH =  tf.placeholder(tf.float32, name="beta")
        if self.n_out==3:
            std=10.
        else:
            std=0.1
            
        if self.cnn is True:
            if  self.n_steps is None:
                self.measPH=tf.placeholder(tf.float32,shape=(None,self.meas_dims[0],self.meas_dims[1],self.n_chan_in), name="meas")
                self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_out), name="qtrue")
            else:
                self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims[0], self.meas_dims[1],self.n_chan_in), name="meas")
                self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_out), name="qtrue")
            # Store layers weight & bias

            with tf.name_scope('convolutional_layer_1'):
                self.wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
                self.bc1 = tf.Variable(tf.random_normal([self.n_conv1]))

                if self.n_steps is None:
                    conv1 = conv2d(self.measPH,self.wc1,self.bc1,name="conv1")
                else:
                    #reshape to fold batch and timestep indices, needed for compatibility with conv2d
                    measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
                    conv1 = conv2d(measfold,self.wc1,self.bc1,name="conv1")

                conv1 = max_pool(conv1,k=self.k_pool)

                # Apply Dropout
                conv1 = tf.nn.dropout(conv1,self.dropoutPH)

            with tf.name_scope('convolutional_layer_2'):
                self.wc2 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_conv1, self.n_conv2])) # kxk conv, 2 input, n_conv outputs
                self.bc2 = tf.Variable(tf.random_normal([self.n_conv2]))
                conv2 = conv2d(conv1,self.wc2,self.bc2,name="conv2" )
                # Apply Dropout
                conv2 = tf.nn.dropout(conv2, self.dropoutPH)

            with tf.name_scope('dense_layer'):
                if self.n_steps is None:
                    self.wd = tf.Variable(tf.random_normal([self.n_dense, self.n_out]),stddev=std) # fully connected, image inputs, n_out outputs
                    self.bd = tf.Variable(tf.random_normal([self.n_out]),stddev=std)
                    dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv2 output to fit dense layer input
                    logits = tf.add(tf.matmul(dense,self.wd),self.bd)#logits

                else:
                    dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
                    self.wd = tf.Variable(tf.truncated_normal([self.n_dense, self.n_lstm], stddev=0.1))
                    self.bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
                    dense_out = tf.nn.softmax(tf.matmul(dense, self.wd) + self.bd,name="dense_out")
                    dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
        else:#NO CNN
            if  self.n_steps is None:
                self.measPH=tf.placeholder(tf.float32,shape=(None,self.meas_dims), name="meas")
                self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_out), name="qtrue")
            else:
                self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims), name="meas")
                self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_out), name="qtrue")

            with tf.name_scope('dense_layer'):
                if self.n_steps is None:
                    self.wd = tf.Variable(tf.random_normal([self.meas_dims, self.n_out]),stddev=std) # fully connected, image inputs, n_out outputs
                    self.bd = tf.Variable(tf.random_normal([self.n_out]),stddev=std)
                    logits = tf.add(tf.matmul(self.measPH,self.wd),self.bd)#logits
                else:
                    dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
                    self.wd = tf.Variable(tf.truncated_normal([self.n_dense, self.n_lstm], stddev=0.1))
                    self.bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
                    dense_out = tf.nn.relu(tf.matmul(dense, self.wd) + self.bd,name="dense_out")#try relu
                    dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
                
        with tf.name_scope('rnn_layer'):
            if self.n_steps is not None:
                #now predict sequence of firing
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  

                data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
                output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

                val = tf.transpose(output,[1,0,2])#nxbxn_lstm
                last = tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
                outs = tf.reshape(output, [-1, self.n_lstm])#n*bxn_lstm
                
                self.wrnn = tf.Variable(tf.random_normal([self.n_lstm,self.n_out], stddev=std))
                self.brnn = tf.Variable(tf.random_normal([self.n_out],stddev=std))

                logits = tf.add(tf.matmul(outs,self.wrnn),self.brnn)#logits - b*nxp
                #logits = tf.reshape(logits,[-1,self.n_out])#b*nxp
                
                logits_last=tf.add(tf.matmul(last,self.wrnn),self.brnn)#logits - bxp

        with tf.name_scope('cost'):
            if self.n_steps is not None:
                if self.n_out==3:
                    self.qhat = logits
                    qhat_last = logits_last                
                else:
                    self.qhat = tf.nn.softmax(logits,name="qhat")
                    qhat_last = tf.nn.softmax(logits_last,name="qhat_last")
            else:
                    if self.n_out==3:
                        self.qhat = logits
                    else:
                        self.qhat = tf.nn.softmax(logits,name="qhat")

            if self.cnn is True:
                reg=tf.multiply(self.betaPH,
                                tf.add(tf.add(tf.nn.l2_loss(self.wd),tf.nn.l2_loss(self.wc1)),
                                       tf.nn.l2_loss(self.wc2)))
            else:
                reg=tf.multiply(self.betaPH,tf.nn.l2_loss(self.wd))
                
            A=tf.argmax(logits,1)
            if self.n_steps is None:
                
                B=tf.argmax(self.qtruePH,1)
                qtrue_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.qtruePH_OH),name="cross"),reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
                self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.qhat,self.qtruePH))),name="rmse"),reg)

                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.qtruePH_OH),name="cross_last"),reg)        
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy_last")
                self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.qhat,self.qtruePH))),name="rmse_last"),reg)
            else:
                qtrue_unflat = tf.reshape(self.qtruePH,[-1,self.n_out])#b*nxp
                AA=tf.argmax(logits_last,1)
                B=tf.argmax(qtrue_unflat,1)#b*nx1
                qtrue_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                
                qtrue_tran = tf.transpose(self.qtruePH,[1,0,2])#nxbxp
                qtrue_last = tf.gather(qtrue_tran, int(qtrue_tran.get_shape()[0])-1)#bxp
                BB=tf.argmax(qtrue_last,1)#bx1
                qtrue_last_OH = tf.one_hot(BB,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                
                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=qtrue_OH),name="cross"),reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
                self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.qhat,qtrue_unflat))),name="rmse"),reg)
                
                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_last, labels=qtrue_last_OH),name="cross_last"),reg)
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(AA,BB),tf.float32),name="accuracy_last")
                self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(qtrue_last,qhat_last))),name="rmse_last"),reg)
            
        with tf.name_scope('train_step'):
            #use rmse as cost function if you are trying to fit current density OR location.
            if self.n_steps is None:
                if self.cost_func=='cross':
                    self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross)
                elif self.cost_func=='rmse':
                    self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse)
                else:
                    self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross)
            else:
                if self.cost_time=='all':
                    if self.cost_func=='cross':
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross)
                    elif self.cost_func=='rmse':
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse)
                    else:
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse)
                elif self.cost_time=='last':
                    if self.cost_func=='cross':
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_last)
                    elif self.cost_func=='rmse':
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse_last)
                    else:
                        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse_last)
                else:
                    self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.rmse_last)
            self.saver = tf.train.Saver()
            
        self.init_step = tf.global_variables_initializer()
