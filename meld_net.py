
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

class meld:
    def __init__(self,learning_rate,meas_dims,k_conv,k_pool,n_chan_in,n_conv1,n_conv2,n_out,n_steps=None,n_lstm=None,n_layer=None, cnn=True, rnn=True):
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
        self.n_steps=n_steps
        print "n_steps: ", self.n_steps
        self.n_lstm=n_lstm
        print "n_lstm: ", self.n_lstm
        self.n_layer=n_layer
        print "n_layer: ", self.n_layer
        self.cnn=cnn
        print self.cnn
        self.rnn=rnn
        print self.rnn
        if self.cnn is True:
            self.n_dense=int((self.meas_dims[0]-self.k_conv+1)/self.k_pool-self.k_conv+1)*int((self.meas_dims[1]-self.k_conv+1)/self.k_pool-self.k_conv+1)*self.n_conv2
        else:
            self.n_dense=self.meas_dims
        print "n_dense: ", self.n_dense
        if self.n_out==3:
            self.std=10.
        else:
            self.std=0.1

    def cnn_nn(self):        
        # Store layers weight & bias
        with tf.name_scope('convolutional_layer_1'):
            self.wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            self.bc1 = tf.Variable(tf.random_normal([self.n_conv1]))
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
            dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_out], stddev=self.std))
            bd = tf.Variable(tf.random_normal([self.n_out], stddev=self.std))
            dense_out = tf.add(tf.matmul(dense, wd),bd)
            self.logits = tf.nn.dropout(dense_out, self.dropoutPH)
            if self.n_out==3:
                self.qhat = self.logits
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
            self.reg=tf.multiply(self.betaPH,
                                 tf.add(tf.add(tf.nn.l2_loss(wd),tf.nn.l2_loss(self.wc1)),
                                        tf.nn.l2_loss(self.wc2)))
            self.A=tf.argmax(self.logits,1)

    def rnn_nn(self):
        with tf.name_scope('dense_layer'):
            dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm], stddev=0.1))
            bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
                
        with tf.name_scope('rnn_layer'):
            #now predict sequence of firing
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  
            
            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

            val = tf.transpose(output,[1,0,2])#nxbxn_lstm
            last = tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output, [-1, self.n_lstm])#n*bxn_lstm
                
            self.wrnn = tf.Variable(tf.random_normal([self.n_lstm,self.n_out], stddev=self.std))
            self.brnn = tf.Variable(tf.random_normal([self.n_out],stddev=self.std))

            self.logits = tf.add(tf.matmul(outs,self.wrnn),self.brnn)#logits - b*nxp
            #logits = tf.reshape(logits,[-1,self.n_out])#b*nxp
                
            self.logits_last=tf.add(tf.matmul(last,self.wrnn),self.brnn)#logits - bxp

            if self.n_out==3:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")
            
            self.reg=tf.multiply(self.betaPH,tf.nn.l2_loss(wd))
            
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)
        
    def cnn_rnn(self):
        # Store layers weight & bias
        with tf.name_scope('convolutional_layer_1'):
            self.wc1 = tf.Variable(tf.random_normal([self.k_conv, self.k_conv, self.n_chan_in, self.n_conv1])) # kxk conv, 2 input, n_conv outputs
            self.bc1 = tf.Variable(tf.random_normal([self.n_conv1]))
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
            dense = tf.reshape(conv2, [-1, self.n_dense]) # Reshape conv1 output to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm], stddev=0.1))
            bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
        with tf.name_scope('rnn_layer'):
            #now predict sequence of firing
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_lstm,state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropoutPH)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.n_layer,state_is_tuple=True)  

            data = tf.reshape(dense_out, [-1, self.n_steps, self.n_lstm])
            output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

            val = tf.transpose(output,[1,0,2])#nxbxn_lstm
            last = tf.gather(val, int(val.get_shape()[0])-1)#bxn_lstm
                
            outs = tf.reshape(output, [-1, self.n_lstm])#n*bxn_lstm
                
            wrnn = tf.Variable(tf.random_normal([self.n_lstm,self.n_out], stddev=self.std))
            brnn = tf.Variable(tf.random_normal([self.n_out],stddev=self.std))

            self.logits = tf.add(tf.matmul(outs,wrnn),brnn)#self.logits - b*nxp
            self.logits_last=tf.add(tf.matmul(last,wrnn),brnn)#logits - bxp

            if self.n_out==3:
                self.qhat = self.logits
                self.qhat_last = self.logits_last                
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
                self.qhat_last = tf.nn.softmax(self.logits_last,name="qhat_last")

            self.reg=tf.multiply(self.betaPH,
                            tf.add(tf.add(tf.nn.l2_loss(wd),tf.nn.l2_loss(self.wc1)),
                                   tf.nn.l2_loss(self.wc2)))
            self.A=tf.argmax(self.logits,1)
            self.AA=tf.argmax(self.logits_last,1)

    def mlp(self):
        with tf.name_scope('input_layer'):
            dense = tf.reshape(self.measPH, [-1, self.meas_dims]) # Reshape input to fit dense layer input
            wd = tf.Variable(tf.random_normal([self.n_dense, self.n_lstm], stddev=0.1))
            bd = tf.Variable(tf.constant(0.1, shape=[self.n_lstm]))
            dense_out = tf.nn.relu(tf.add(tf.matmul(dense, wd),bd),name="dense_out")#try relu
            dense_out = tf.nn.dropout(dense_out, self.dropoutPH)
        with tf.name_scope('output_layer'):
            wo = tf.Variable(tf.random_normal([self.n_lstm, self.n_out], stddev=self.std))
            bo = tf.Variable(tf.random_normal([self.n_out], stddev=self.std))
            self.logits = tf.add(tf.matmul(dense_out, wo),bo)
            if self.n_out==3:
                self.qhat = self.logits
            else:
                self.qhat = tf.nn.softmax(self.logits,name="qhat")
            
            self.reg=tf.add(tf.multiply(self.betaPH,tf.nn.l2_loss(wd)),tf.multiply(self.betaPH,tf.nn.l2_loss(wo)))
            
            self.A=tf.argmax(self.logits,1)

    def network(self):
        self.dropoutPH = tf.placeholder(tf.float32, name="dropout")
        self.betaPH =  tf.placeholder(tf.float32, name="beta")
        if self.cnn is False:
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims), name="meas")
        else:
            self.measPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.meas_dims[0], self.meas_dims[1],self.n_chan_in), name="meas")

        if self.rnn is True:
            if self.cnn is True:
                self.cnn_rnn()
            else:
                self.rnn_nn()
        else:            
            if self.cnn is True:
                self.cnn_nn()
            else:
                self.mlp()          

    def trainer(self):
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.saver = tf.train.Saver()
            
    def initializer(self):
        self.init_step = tf.global_variables_initializer()
        
    def cost(self): 
                    
        self.qtrainPH=tf.placeholder(tf.float32,shape=(None, self.n_steps, self.n_out), name="qtrain")
        qtrain_unflat = tf.reshape(self.qtrainPH,[-1,self.n_out])#b*nxp

        with tf.name_scope('cost'):
            if self.rnn is False:                
                B=tf.argmax(qtrain_unflat,1)
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
        
                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, qtrain_OH),name="cross"),self.reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.A,B),tf.float32),name="accuracy")
                self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,qtrain_unflat))),name="rmse"),self.reg)

                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, qtrain_OH),name="cross_last"),self.reg)        
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(self.A,B),tf.float32),name="accuracy_last")
                self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,self.qtrainPH))),name="rmse_last"),self.reg)
                if self.n_out==3:
                    self.cost = self.rmse 
                else:
                    self.cost = self.cross

            else:
                B=tf.argmax(qtrain_unflat,1)#b*nx1
                qtrain_OH = tf.one_hot(B,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
        
                qtrain_tran = tf.transpose(self.qtrainPH,[1,0,2])#nxbxp
                qtrain_last = tf.gather(qtrain_tran, int(qtrain_tran.get_shape()[0])-1)#bxp
                BB=tf.argmax(qtrain_last,1)#bx1
                qtrain_last_OH = tf.one_hot(BB,self.n_out,on_value=1,off_value=0,axis=-1)#need one-hot for CE calculation
                
                self.cross = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, qtrain_OH),name="cross"),self.reg)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.A,B),tf.float32),name="accuracy")
                self.rmse = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.qhat,qtrain_unflat))),name="rmse"),self.reg)
                
                self.cross_last = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits_last, qtrain_last_OH),name="cross_last"),self.reg)
                self.accuracy_last = tf.reduce_mean(tf.cast(tf.equal(self.AA,BB),tf.float32),name="accuracy_last")
                self.rmse_last = tf.add(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(qtrain_last,self.qhat_last))),name="rmse_last"),self.reg)
                if self.n_out==3:
                    self.cost = self.rmse_last 
                else:
                    self.cost = self.cross_last

            with tf.name_scope('summaries'):
                self.train_summary = tf.summary.scalar('mean/train_cost', self.cost)
                self.valid_summary = tf.summary.scalar('mean/valid_cost', self.cost)