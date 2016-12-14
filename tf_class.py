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
    def __init__(self,learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=None,n_lstm=None,n_layer=None):
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
        self.n_dense=int((self.meas_dims[0]-self.k_conv+1)/self.k_pool-self.k_conv+1)*int((self.meas_dims[1]-self.k_conv+1)/self.k_pool-self.k_conv+1)*self.n_conv2
        print "n_dense: ", self.n_dense
        if n_steps is 1:
            self.n_steps=None
        else:
            self.n_steps=n_steps
        print "n_steps: ", self.n_steps
        self.n_lstm=n_lstm
        print "n_lstm: ", self.n_lstm
        self.n_layer=n_layer
        print "n_layer: ", self.n_layer
        
    def network(self):
        
        self.dropoutPH = tf.placeholder(tf.float32, name="dropout")

        if  self.n_steps is None:
            self.measPH=tf.placeholder(tf.float32,shape=(None,self.meas_dims[0],self.meas_dims[1],self.n_chan_in), name="meas")
            self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_out), name="qtrue")
        else:
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims[0], self.meas_dims[1],self.n_chan_in), name="meas")
            self.qtruePH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_out), name="qtrue")
        # Store layers weight & bias

        with tf.name_scope('convolutional_layer_1'):
            wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            bc1 = tf.Variable(tf.random_normal([self.n_conv1]))

            if self.n_steps is None:
                conv1 = conv2d(self.measPH,wc1,bc1,name="conv1")
            else:
                #reshape to fold batch and timestep indices, needed for compatibility with conv2d
                measfold = tf.reshape(self.measPH,[-1,self.meas_dims[0], self.meas_dims[1], self.n_chan_in])
                conv1 = conv2d(measfold,wc1,bc1,name="conv1")
                
            conv1 = max_pool(conv1,k=self.k_pool)

            # Apply Dropout
            conv1 = tf.nn.dropout(conv1,self.dropoutPH)

        with tf.name_scope('convolutional_layer_2'):
            wc2 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_conv1, self.n_conv2])) # kxk conv, 2 input, n_conv outputs
            bc2 = tf.Variable(tf.random_normal([self.n_conv2]))
            conv2 = conv2d(conv1,wc2,bc2,name="conv2" )
            # Apply Dropout
            conv2 = tf.nn.dropout(conv2, self.dropoutPH)

        with tf.name_scope('dense_layer'):
            if self.n_steps is None:
                wd = tf.Variable(tf.random_normal([self.n_dense, self.n_out])) # fully connected, image inputs, n_out outputs
                bd = tf.Variable(tf.random_normal([self.n_out]))
                dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv2 output to fit dense layer input
                logits = tf.add(tf.matmul(dense,wd),bd)#logits
                self.qhat = tf.nn.softmax(logits,name="qhat")
            else:
                dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
                wd = tf.Variable(tf.truncated_normal([self.n_dense, self.n_lstm], stddev=0.1))
                bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
                dense_out = tf.nn.softmax(tf.matmul(dense, wd) + bd,name="dense_out")
                dense_out = tf.nn.dropout(dense_out, self.dropoutPH)

        with tf.name_scope('rnn_layer'):
            if self.n_steps is not None:
                #now predict sequence of firing
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  

                data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
                output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

                val = tf.transpose(output,[1,0,2])
                last = tf.gather(val, int(val.get_shape()[0])-1)
                
                outs = tf.reshape(output, [-1, self.n_lstm])

                weight = tf.Variable(tf.truncated_normal([self.n_lstm,self.n_out], stddev=0.1))
                bias = tf.Variable(tf.constant(0.1, shape=[self.n_out]))
                
                wrnn = tf.Variable(tf.truncated_normal([self.n_lstm,self.n_out], stddev=0.1))
                brnn = tf.Variable(tf.constant(0.1, shape=[self.n_out]))

                logits = tf.add(tf.matmul(outs,wrnn),brnn)#logits
                logits = tf.reshape(logits,[-1,self.n_out])
                
                logits_last=tf.add(tf.matmul(last,weight),bias)#logits

                qhat_last = tf.nn.softmax(logits_last,name="qhat_last")
                self.qhat = tf.nn.softmax(logits,name="qhat")
                
        with tf.name_scope('cost'):
            A=tf.argmax(logits,1)
            if self.n_steps is None:        
                B=tf.argmax(self.qtruePH,1)
                self.cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.qtruePH),name="cross")
                self.cross_last = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.qtruePH),name="cross_last")

                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
                self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,self.qtruePH))),name="rmse")
            
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy_last")
                self.rmse_last = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,self.qtruePH))),name="rmse_last")
            else:
                AA=tf.argmax(logits_last,1)
                B=tf.argmax(tf.reshape(self.qtruePH,[-1,self.n_out]),1)

                qtrue_tran = tf.transpose(self.qtruePH,[1,0,2])
                qtrue_last = tf.gather(qtrue_tran, int(qtrue_tran.get_shape()[0])-1)

                BB=tf.argmax(qtrue_last,1)
                
                self.cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.reshape(self.qtruePH,[-1,self.n_out])),name="cross")
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(A,B),tf.float32),name="accuracy")
                self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,tf.reshape(self.qtruePH,[-1,self.n_out])))),name="rmse")
                
                self.cross_last = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_last, qtrue_last),name="cross_last")
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(AA,BB),tf.float32),name="accuracy_last")
                self.rmse_last = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qtrue_last,qhat_last))),name="rmse_last")
            
        with tf.name_scope('train_step'):
            #use rmse as cost function if you are trying to fit current density.
            if self.n_steps is None:
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross)
            else:
                self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross)
                #change to cross_last if you want.

        self.init_step = tf.initialize_all_variables()
